import os
import random

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from warmup_scheduler import GradualWarmupScheduler
from utils import get_train_valid_dataset_with_ray
from cfg import CFG
import h5py
import utils
import json
from unetr_pp.network_architecture.acdc.unetr_pp_acdc import UNETR_PP
from transformers import SegformerForSemanticSegmentation
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return loggers


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


def make_dirs(cfg):
    for dir in [cfg.model_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug


class CustomDataset(Dataset):
    def __init__(self, h5_dataset: utils.HDF5CacheResult, is_valid, cfg, transform=None):
        self.dataset = h5_dataset
        self.transform = transform
        self.is_valid = is_valid
        self.cfg = cfg
        with h5py.File(h5_dataset.file_path, 'r') as f:
            self.len = len(f[utils.DatasetNames.TRAIN_IMAGES.value]) if not is_valid else len(
                f[utils.DatasetNames.VALID_IMAGES.value])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.dataset.file_path, 'r') as f:
            # Is validation data
            if self.is_valid:
                image = f[utils.DatasetNames.VALID_IMAGES.value][idx]
                label = f[utils.DatasetNames.VALID_INK_MASKS.value][idx]
                xy = f[utils.DatasetNames.VALID_XY_XYS.value][idx]
                val_segment_idx = f[utils.DatasetNames.VALID_SEGMENT_IDXS.value][idx]
                if self.transform:
                    data = self.transform(image=image, mask=label)
                    image = data['image'].unsqueeze(0)
                    label = data['mask']
                return image, label, xy, val_segment_idx

            # Is not validation data
            else:
                image = f[utils.DatasetNames.TRAIN_IMAGES.value][idx]
                label = f[utils.DatasetNames.TRAIN_INK_MASKS.value][idx]
                if self.transform:
                    data = self.transform(image=image, mask=label)
                    image = data['image'].unsqueeze(0)
                    label = data['mask']
                return image, label


# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

class RegressionPLModel(pl.LightningModule):
    def __init__(self, train_segment_ids, val_segment_ids, stride, size, depth, train_batch_size=None, val_batch_size=None, val_segment_id_shapes=None, output_path=None):
        super(RegressionPLModel, self).__init__()
        self.train_segment_ids = train_segment_ids
        self.val_segment_ids = val_segment_ids
        self.stride = stride
        self.size = size
        self.depth = depth
        self.train_batch_size = CFG.train_batch_size if train_batch_size is None else train_batch_size
        self.val_batch_size = CFG.valid_batch_size if val_batch_size is None else val_batch_size
        self.val_segment_id_shapes = val_segment_id_shapes
        self.output_path = output_path

        self.save_hyperparameters()
        # self.mask_pred = np.zeros(self.hparams.pred_shape)
        # self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_funcs = [(smp.losses.DiceLoss(mode='binary'), smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25))
                           for _ in range(1)]
        self.losses = [
            lambda x, y: 0.5 * self.loss_funcs[i][0](x, y) + 0.5 * self.loss_funcs[i][1](x, y) for i in range(1)
        ]
        # OUT SIZE 3 [torch.Size([8, 1, 32, 256, 256]), torch.Size([8, 1, 32, 64, 64]), torch.Size([8, 1, 16, 32, 32])]
        # So scale factors are 1, 4, 8
        self.scale_factors = [1, 4, 8]

        # self.stn = ThinPlateSplineSTN3D(image_height=CFG.size, image_width=CFG.size, image_depth=CFG.in_channels)
        print("Trying to set up backbone")
        self.backbone = UNETR_PP(in_channels=1,
                                 out_channels=32,
                                 feature_size=16,
                                 num_heads=4,
                                 depths=[3, 3, 3, 3],
                                 dims=[32, 64, 128, 256],
                                 dropout_rate=0.2,
                                 do_ds=False,
                                 window_size=size,
                                 depth=depth)
        # First pooling layer: Reduce depth by a factor of 16
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", num_labels=1,
                                                                          ignore_mismatched_sizes=True, num_channels=32)
        self.segformer.train()
        self.upscale1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscale2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        # print(x.shape)
        # torch.Size([256, 1, 30, 64, 64])
        # 256=batch size
        # 1 channel input for i3d
        # 30 channel actual input image
        # 64x64 patch
        # x = self.stn(x)
        # [8b, 1c, 30d, 256h, 256w]
        # x = x.permute(0, 1, 3, 4, 2)
        # [8b, 1c, 256h, 256w, 30d]
        x = self.backbone(x).max(axis=2)[0]
        # OUT SIZE 3 [torch.Size([8, 1, 32, 256, 256]), torch.Size([8, 1, 32, 64, 64]), torch.Size([8, 1, 16, 32, 32])]
        # print(f"OUT SIZE {len(x)} {[t.shape for t in x]}")
        # x = torch.sum(torch.stack(x, dim=0), dim=0)
        x = self.segformer(x).logits
        x = self.upscale1(x)
        x = self.upscale2(x)
        # [8b, 1c, 30d, 256h, 256w]
        # x = x.permute(0, 1, 4, 2, 3)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.losses[0](outputs.squeeze(), y.squeeze())
        # for i, loss_func in enumerate(self.losses):
        #     label = F.interpolate(y, scale_factor=1 / self.scale_factors[i]).squeeze()
        #     if loss is None:
        #         loss = loss_func(outputs[i].squeeze(), label)
        #     else:
        #         loss += loss_func(outputs[i].squeeze(), label)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train_total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys, segment_id_idx = batch
        outputs = self(x)
        loss = self.losses[0](outputs.squeeze(), y.squeeze())
        # for i, loss_func in enumerate(self.losses):
        #     label = F.interpolate(y, scale_factor=1 / self.scale_factors[i]).squeeze()
        #     if loss is None:
        #         loss = loss_func(outputs[i].squeeze(), label)
        #     else:
        #         loss += loss_func(outputs[i].squeeze(), label)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("val_total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        xs, ys, xyxys, segment_id_idxs = batch
        outputs = self(xs)
        return {"preds": outputs, "xyxys": xyxys, "segment_id_idxs": segment_id_idxs}

    def on_test_end(self, outputs) -> None:
        results = []
        mask_counts = []
        kernel = gkern(self.size, 1)
        kernel = kernel.max()
        print("Writing results")
        for shape in self.val_segment_id_shapes:
            results.append(np.zeros(shape))
            mask_counts.append(np.zeros(shape))
        for out in outputs:
            preds = out["preds"]
            xyxys = out["xyxys"]
            segment_id_idxs = out["segment_id_idxs"]
            for i, id_idx in segment_id_idxs:
                x1, y1, x2, y2 = xyxys[i]
                mask_pred = preds[i]
                mask_count = mask_counts[i]
                results[i][y1:y2, x1:x2] += np.multiply(
                    F.interpolate(mask_pred.unsqueeze(0).float(), scale_factor=1, mode='bilinear').squeeze(0).squeeze(
                        0).numpy(), kernel)
                mask_count[y1:y2, x1:x2] += np.ones((self.size, self.size))
        for i, _ in enumerate(results):
            results[i] /= mask_counts[i]
            results[i] = np.clip(np.nan_to_num(results[i]), a_min=0, a_max=1)
            results[i] /= results[i].max()
            results[i] = (results[i] * 255).astype(np.uint8)
            img = Image.fromarray(results[i])
            img.save(os.path.join(self.output_path, f"{self.val_segment_ids[i]}_inklabels.png"))

    def on_validation_epoch_end(self):
        # self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred),
        #                            where=self.mask_count != 0)
        # wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])
        #
        # # reset mask
        # self.mask_pred = np.zeros(self.hparams.pred_shape)
        # self.mask_count = np.zeros(self.hparams.preds_shape)
        return None

    def configure_optimizers(self):

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)

        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        result: utils.HDF5CacheResult = get_train_valid_dataset_with_ray(
            self.train_segment_ids,
            self.val_segment_ids,
            self.stride, self.size, self.depth)

        train_dataset = CustomDataset(
            h5_dataset=result, is_valid=False, transform=get_transforms(data='train', cfg=CFG), cfg=CFG)

        return DataLoader(train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True, pin_memory=True, drop_last=True, num_workers=CFG.num_workers
                          )

    def val_dataloader(self):
        result: utils.HDF5CacheResult = get_train_valid_dataset_with_ray(
            self.train_segment_ids,
            self.val_segment_ids,
            self.stride, self.size, self.depth)
        valid_dataset = CustomDataset(h5_dataset=result, is_valid=True, transform=get_transforms(data='valid', cfg=CFG),
                                      cfg=CFG)
        return DataLoader(valid_dataset,
                          batch_size=self.val_batch_size, num_workers=CFG.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        result: utils.HDF5CacheResult = get_train_valid_dataset_with_ray(
            self.train_segment_ids,
            self.val_segment_ids,
            self.stride, self.size, self.depth)
        test_dataset = CustomDataset(h5_dataset=result, is_valid=True, transform=get_transforms(data='valid', cfg=CFG),
                                      cfg=CFG)
        return DataLoader(test_dataset,
                          batch_size=self.val_batch_size, num_workers=CFG.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True)


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
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 30, eta_min=1e-4)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler_cosine)

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


if __name__ == '__main__':
    # valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)

    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    unused: utils.HDF5CacheResult = get_train_valid_dataset_with_ray(
        CFG.train_segment_ids,
        CFG.val_segment_ids,
        CFG.stride, CFG.size, 16)
    print(f"Going to train on train set: {CFG.train_segment_ids}")
    print(f"Going to train on validation set: {CFG.val_segment_ids}")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    enc = 'unetr_pp'
    run_slug = f'v{metadata["version"]}_{metadata["type"]}_{CFG.size}x{CFG.size}_{enc}'
    wandb_logger = WandbLogger(project="Vesuvius-Repro", name=run_slug, group='mega-cluster')

    model = RegressionPLModel(CFG.train_segment_ids, CFG.val_segment_ids, CFG.stride, CFG.size, CFG.in_channels)

    wandb_logger.watch(model, log="all", log_freq=100)
    model_path = os.path.join(CFG.model_dir, f"v{metadata['version']}_{CFG.size}_{metadata['type']}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    devices = list(range(0, 5))
    devices.extend(list(range(11, 16)))
    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="gpu",
        devices=devices,
        strategy='ddp',
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[ModelCheckpoint(filename='{val_total_loss:.4f}-{train_total_loss:.4f}-{epoch}',
                                   dirpath=model_path,
                                   monitor='val_total_loss', mode='min', save_top_k=5),
                   LearningRateMonitor(logging_interval='step')],
        log_every_n_steps=5
    )
    trainer.fit(model=model)

    wandb.finish()
