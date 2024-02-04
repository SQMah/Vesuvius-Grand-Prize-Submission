import os
import random

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import scipy.stats as st
import segmentation_models_pytorch as smp
import torch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from albumentations.pytorch import ToTensorV2
from tap import Tap
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from warmup_scheduler import GradualWarmupScheduler
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.cuda.amp import autocast
import math
import utils
import cfg
from accelerate import Accelerator
import gc
from training import RegressionPLModel


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


class InferenceArgumentParser(Tap):
    segment_id: str = '20230925002745'
    segment_path: str = './eval_scrolls'
    # EDIT
    batch_size: int = 64
    size: int = 256
    stride: int = 32
    model_path: str = 'training/v10_256_greybox_invert_new_aug/val_total_loss=0.5675-train_total_loss=0.5765-epoch=11.ckpt'
    out_path: str = './results/v10_size256_stride64_greybox_invert_new_aug'
    # DONE_EDIT
    start_idx: int = 15
    workers: int = 23
    reverse: int = 0


args = InferenceArgumentParser().parse_args()


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'

    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'

    in_chans = 30  # 65
    encoder_depth = 5
    # ============== training cfg =============
    size = args.size
    stride = args.stride

    train_batch_size = args.batch_size  # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 50  # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 2

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 5

    print_freq = 50
    num_workers = args.workers

    seed = 42


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

def lcm(x, y):
    return abs(x * y) // math.gcd(x, y)


def read_image_mask(fragment_id, start_idx=15, end_idx=45, rotation=0):
    dataset_path = args.segment_path
    idxs = range(start_idx, end_idx)

    # Compute the padding values outside the process_image function
    sample_image = cv2.imread(f"{dataset_path}/{fragment_id}/layers/{start_idx:02}.tif", 0)
    print(f"Sample image shape {sample_image.shape}")
    pad0 = (CFG.size - sample_image.shape[0]) % CFG.size
    pad1 = (CFG.size - sample_image.shape[1]) % CFG.size
    print(f"{pad0}, {pad1}")

    def process_image(i):
        print(f"{dataset_path}/{fragment_id}/layers/{i:02}.tif")
        image = cv2.imread(f"{dataset_path}/{fragment_id}/layers/{i:02}.tif", cv2.IMREAD_GRAYSCALE)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)
        return image

    # Using ThreadPoolExecutor to parallelize image reading and processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        images = list(executor.map(process_image, idxs))

    print(f"Total len {len(images)}")
    images = np.stack(images, axis=2)
    print(f"Images shape {images.shape}")

    print("Done reading image layers")

    fragment_mask = None
    if os.path.exists(f'{dataset_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask = cv2.imread(CFG.comp_dataset_path + f"{dataset_path}/{fragment_id}/{fragment_id}_mask.png", cv2.IMREAD_GRAYSCALE)
        print(f"Fragment mask shape {fragment_mask.shape}")
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        kernel = np.ones((16, 16), np.uint8)
        fragment_mask = cv2.erode(fragment_mask, kernel, iterations=1)
    print("Done reading image masks")
    return images, fragment_mask


def extract_tiles_chunk(chunk, image, fragment_mask, process_num):
    print(f"Chunk process {process_num}, image {image.shape}, fragment_mask {fragment_mask.shape}")
    with ThreadPoolExecutor(max_workers=1) as thread_executor:
        desc = f"Process {process_num}"
        # Adjusting the lambda to pass image and fragment_mask for each task
        res = thread_executor.map(lambda yx: extract_tile(yx[0], yx[1], image, fragment_mask),
                                        tqdm(chunk, desc=desc, leave=False))
    return list(res)


def extract_tile(y1, x1, image, fragment_mask):
    y2 = y1 + CFG.size
    x2 = x1 + CFG.size
    try:
        if not np.any(fragment_mask[y1:y2, x1:x2] == 0):
            return image[y1:y2, x1:x2], [x1, y1, x2, y2]
    except IndexError as e:
        print(f"Errored out {x1, x2, y1, y2} frag {fragment_mask.shape} image {image.shape}")
        raise e
    return None, None


def get_img_splits(fragment_id, s, e, rotation=0):
    print("Creating tiles")
    images = []
    xyxys = []

    image, fragment_mask = read_image_mask(fragment_id, s, e, rotation)
    print(f"Received shapes image {image.shape}, fragment {fragment_mask.shape}")
    x1_list = list(range(0, image.shape[1] - CFG.size + 1, CFG.stride))
    y1_list = list(range(0, image.shape[0] - CFG.size + 1, CFG.stride))

    tasks = [(y1, x1) for y1 in y1_list for x1 in x1_list]

    # Divide tasks into chunks
    chunk_size = len(tasks) // CFG.num_workers
    print(f"Num tasks {len(tasks)} chunk size {chunk_size}")
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    image_iter, fragment_iter = [image for _ in range(CFG.num_workers)], [fragment_mask for _ in range(CFG.num_workers)]
    with ThreadPoolExecutor(max_workers=CFG.num_workers) as executor:
        results_chunked = list(executor.map(extract_tiles_chunk, chunks, image_iter, fragment_iter, range(CFG.num_workers)))

    # Flatten the results
    results = [item for sublist in results_chunked for item in sublist]

    for res_img, res_xyxy in results:
        if res_img is not None:
            images.append(res_img)
            xyxys.append(res_xyxy)
    print("Done processing tile results")

    test_dataset = CustomDatasetTest(images, np.stack(xyxys), CFG, transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean=[0] * CFG.in_chans,
            std=[1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))
    print("Done creating test dataset.")

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.valid_batch_size,
                             shuffle=False,
                             num_workers=23, pin_memory=True, persistent_workers=True, drop_last=False,
                             )
    return test_loader, np.stack(xyxys), (image.shape[0], image.shape[1]), fragment_mask


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image, xy


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')


from collections import OrderedDict


def normalization(x):
    """input.shape=(batch,f1,f2,...)"""
    # [batch,f1,f2]->dim[1,2]
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + 1e-9)

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


def TTA(x: tc.Tensor, model: nn.Module):
    # x.shape=(batch,c,h,w)
    shape = x.shape
    x = [x, *[tc.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)], ]
    x = tc.cat(x, dim=0)
    x = model(x)
    # x=torch.sigmoid(x)
    # print(x.shape)
    x = x.reshape(4, shape[0], CFG.size // 4, CFG.size // 4)

    x = [tc.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = tc.stack(x, dim=0)
    return x.mean(0)


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    model.eval()
    kernel = gkern(CFG.size, 1)
    kernel = kernel / kernel.max()
    for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        xys_torch = torch.Tensor(xys)
        with torch.no_grad():
            y_preds = model(images)
            # y_preds =TTA(images,model)
        # y_preds = y_preds.to('cpu').numpy()
        y_preds = accelerator.gather(y_preds)
        xys = accelerator.gather(xys_torch)
        if accelerator.is_main_process:
            y_preds = torch.sigmoid(y_preds).to('cpu')
            xys = xys.to('cpu')
            for i, (x1, y1, x2, y2) in enumerate(xys):
                mask_pred[y1:y2, x1:x2] += np.multiply(
                    F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=4, mode='bilinear').squeeze(0).squeeze(
                        0).numpy(), kernel)
                mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    if accelerator.is_main_process:
        mask_pred /= mask_count
        # mask_pred/=mask_pred.max()
        return mask_pred


if __name__ == "__main__":
    accelerator = Accelerator()
    model = RegressionPLModel.load_from_checkpoint(args.model_path, strict=False)
    device = accelerator.device
    accelerator.print("Done loading model!")
    model = accelerator.prepare_model(model)
    model.eval()
    accelerator.print("Done with eval!")
    for fragment in cfg.CFG.test_segment_ids:
        if not os.path.isdir(os.path.join(args.segment_path, fragment)):
            continue
        if not os.path.isdir(args.out_path):
            try:
                os.mkdir(args.out_path)
            except:
                pass
        fragment_id = fragment
        out_save = f'{args.out_path}/{fragment_id}_inklabels.png'
        if os.path.exists(out_save):
            accelerator.print(f"Skipping {fragment}")
            continue
        accelerator.print(f"Processing fragment {fragment}")
        try:
            test_loader, test_xyxz, test_shape, fragment_mask = get_img_splits(fragment_id, args.start_idx, args.start_idx + 30,
                                                                               0)
            test_loader = accelerator.prepare_data_loader(test_loader)
            accelerator.print("Done getting splits!")
            accelerator.print("Running predict function")
            with autocast(True):
                mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
                accelerator.print("Done with predict function")
                del test_loader
                del test_xyxz
                del test_shape
                del fragment_mask
                accelerator._dataloaders = []
                gc.collect()
                if accelerator.is_main_process:
                    mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
                    mask_pred /= mask_pred.max()
                    mask_pred = (mask_pred * 255).astype(np.uint8)
                    mask_pred = Image.fromarray(mask_pred)
                    mask_pred.save(out_save)
        except Exception as e:
            accelerator.print(f"Failed handling fragment {fragment_id} with exception {e}")
            continue