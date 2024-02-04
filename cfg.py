import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from custom_augmentations import ChannelInvert, FourthAugment
from segment_utils import get_segment_id_paths_dict
import torch
import cv2

# "20230827161847", "20230925090314", "20230922174128", "20230925002745" are all very similar fragments
# "20231005123333" also contains 'purple but is more expansive'
segment_id_data_paths = get_segment_id_paths_dict('./data')
val_segment_ids = ['20231007101619', '20231005123336', '20231031143852']


# val_segment_ids = ["20230827161847"]


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
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone = 'resnet3d'
    in_channels = 16  # 65
    # ============== training cfg =============
    # size = 64
    # stride = 32
    size = 256
    stride = 64
    # Train batch size:
    # 256 for 64
    # 16 for 256
    # 8 for 512
    train_batch_size = 8
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 120

    # adamW warmupあり
    warmup_factor = 10
    # 64: 2e-5
    # 256: 5e-5
    # 512: 1e-4
    lr = 5e-5
    # ============== fold =============
    valid_id = '20230827161847'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== model training =============
    model_dir = "./training"

    # ============== fixed =============
    num_workers = max((os.cpu_count() - 1 - torch.cuda.device_count()) // 2, 2)

    seed = 0

    # ============== set dataset path =============

    """{'20231005123335': 252872658.0, '20231012184421': 173507978.5, '20231022170900': 155218812.0,
    '20230702185753': 154629633.5, '20231106155351': 151181172.5, '20231012173610': 142181724.5, '20231007101615': 122819983.0, '20231031143852': 121577337.5,
    '20231106155350': 94503236.0, '20230701020044': 92780785.0, '20231012184420':
    83153569.5, '20231016151000': 75713336.5, '20231012085431': 60908611.0, '20231024093300': 58613658.5,
    '20230522181603': 48072778.0, '20230925002745': 41334896.0, '20230925090314': 40987711.5, '20231001164029':
    34019414.5, '20230827161847': 32744247.0, '20230922174128': 31996743.5, '20230903193206': 28214956.0,
    '20231004222109': 26136148.0, '20231011111857': 24486950.0, '20231011144857': 23821611.0, '20230904020426':
    23579711.0, '20230613204956': 23302226.0, '20230926164631': 23049179.5, '20230926164853': 21647513.5,
    '20230905134255': 21588509.5, '20230909121925': 21149985.0, '20230601192025': 20536616.0, '20230509182749':
    20004406.5, '20230709155141': 19541563.0, '202305101530026': 19204888.0, '20230901184804': 17781433.5,
    '20230826211400': 16889650.0, '20230904135535': 16753744.0, '20230919113918': 16730017.5, '20230601193301':
    16677788.0, '20230625171244': 16638574.0, '20230629215956': 15774059.0, '20230522215721': 14959700.5,
    '20230620230617': 14252210.5, '20230820203112': 14230582.0, '20230826170124': 14105868.0, '20230620230619':
    14090848.0, '20230531211425': 13838306.0, '20230531193658': 13086224.0, '20230520175435': 13009565.0,
    '20230713152725': 12775564.5, '20230627122904': 12537316.5, '20230511215040': 12405763.5, '20230901234823':
    12338239.5, '20230902141231': 12179544.0, '20230527020406': 11883676.5, '20230627170800': 11628297.5,
    '20230530172803': 11279499.0, '20230627202005': 11180858.5, '20230516115453': 10409651.5, '20230820174948':
    10409295.0, '20230701115953': 9967918.0, '20230512123446': 9788895.5, '20230711201157': 9451719.0,
    '20230526154635': 9385846.5, '20230826135043': 8909695.5, '20230602213452': 8829000.5, '20230605065957':
    8775821.5, '20230619113941': 8500759.5, '20230606222610': 8456403.5, '20230624144604': 8399360.0,
    '20230526183725': 8365712.5, '20230611145109': 8355724.5, '20230509173534': 8092302.0, '20230626140105':
    8092271.0, '20230523002821': 7964972.0, '20230531101257': 7804824.0, '20230530212931': 7715629.0,
    '20230601201143': 7632988.0, '20230702182347': 7580059.0, '20230624160816': 7518020.0, '20230828154913':
    7302046.0, '20230619163051': 7182596.5, '20230601204340': 7094981.0, '20230530164535': 7044125.5,
    '20230603153221': 6980561.5, '20230612195231': 6842359.0, '20230621182552': 6824089.0, '20230819210052':
    6712093.5, '20230426144221': 6696410.5, '20230608200454': 6668481.0, '20230518130337': 6400431.5,
    '20230812170020': 6344088.5, '20230522152820': 6257652.0, '20230606105130': 6245804.5, '20230709211458':
    6209598.0, '20230521113334': 6189495.0, '20230608150300': 6153048.5, '20230504093154': 6107879.5,
    '20230813_frag_2': 6053496.0, '20230819093803': 5949111.0, '20230820091651': 5831808.0, '20230522182853':
    5781090.5, '20230519215753': 5437746.0, '20230621122303': 5393038.5, '20230613144727': 5298101.0,
    '20230604111512': 5271855.0, '20230623160629': 5138164.5, '20230706165709': 5053031.0, '20230801194757':
    4886183.0, '20230717092556': 4756056.0, '20230604112252': 4749531.5, '20230528112855': 4696392.5,
    '20230721143008': 4693181.0, '20230624190349': 4678521.5, '20230608222722': 4638521.5, '20230611014200':
    4638521.5, '20230531121653': 4635687.5, '20230604161948': 4633874.0, '20230517205601': 4551878.0,
    '20230808163057': 4457237.5, '20230707113838': 4406631.5, '20230705142414': 4246641.5, '20230530025328':
    4041419.5, '20230515151114': 4013265.5, '20230625194752': 3950837.0, '20230425200944': 3873007.5,
    '20230505141722': 3770289.5, '20230521114306': 3769835.5, '20230623123730': 3728880.5, '20230529203721':
    3728756.5, '20230626151618': 3704679.5, '20230505113642': 3656960.5, '20230519195952': 3652998.5,
    '20230806132553': 3512335.5, '20230518210035': 3501280.0, '20230509160956': 3376977.0, '20230523043449':
    3337476.0, '20230526015925': 3280348.5, '20230813_real_1': 3230336.0, '20230524173051': 3202969.0,
    '20230801193640': 3138915.0, '20230525212209': 3134729.5, '20230609123853': 3029703.0, '20230806094533':
    3017692.0, '20230719103041': 3000123.0, '20230712124010': 2918603.0, '20230422213203': 2898643.5,
    '20230526175622': 2779751.0, '20230719214603': 2681670.0, '20230521155616': 2587041.5, '20230523191325':
    2560966.5, '20230525200512': 2550635.5, '20230426114804': 2537813.0, '20230525121901': 2469859.5,
    '20230424181417': 2447542.0, '20230526002441': 2407370.0, '20230526205020': 2388430.0, '20230516154633':
    2305804.0, '20230521182226': 2194605.5, '20230515162442': 2112064.0, '20230525190724': 2104768.5,
    '20230621111336': 2101698.0, '20230518181521': 2099079.0, '20230523233708': 2090625.5, '20230519031042':
    2083800.0, '20230711222033': 2031978.0, '20230504171956': 1974785.5, '20230504231922': 1900486.0,
    '20230504151750': 1870881.0, '20230523182629': 1851350.5, '20230524200918': 1834628.5, '20230521193032':
    1714643.5, '20230712210014': 1669836.5, '20230503120034': 1662914.5, '20230918145743': 1640225.5,
    '20230918143910': 1628217.0, '20230918024753': 1627328.0, '20230918140728': 1627195.0, '20230918023430':
    1622627.5, '20230918022237': 1611632.0, '20230918021838': 1598075.0, '20230602092221': 1564238.5,
    '20230518223227': 1500762.0, '20230526164930': 1479885.5, '20230518191548': 1465994.0, '20230522210033':
    1379036.5, '20230517214715': 1377383.0, '20230711210222': 1374260.0, '20230521093501': 1340612.0,
    '20230524005636': 1243710.0, '20230508032834': 1207557.0, '20230424213608': 1207256.5, '20230512112647':
    1190804.0, '20230421235552': 1146700.5, '20230501042136': 1139619.0, '20230721122533': 1139008.5,
    '20230520080703': 1088444.5, '20230525234349': 1046814.5, '20230518012543': 1046559.5, '20230425163721':
    1000584.5, '20230511211540': 997690.0, '20230519140147': 991134.5, '20230513164153': 959083.5, '20230520132429':
    945516.0, '20230517000306': 942057.5, '20230511094040': 940302.5, '20230505093556': 938597.5, '20230516114341':
    925847.0, '20230517193901': 914622.5, '20230521104548': 914600.5, '20230524163814': 806040.5, '20230507064642':
    782553.0, '20230508171353': 778401.0, '20230516112444': 738724.0, '20230525194033': 717009.0, '20230522055405':
    681643.0, '20230512123540': 677859.0, '20230520191415': 669377.5, '20230525051821': 640381.0, '20230519202000':
    627765.0, '20230720215300': 618186.0, '20230504225948': 563100.5, '20230512192835': 527279.0, '20230505175240':
    511051.0, '20230524092434': 486739.5, '20230522151031': 462399.5, '20230511204029': 443238.0, '20230422011040':
    432979.0, '20230506133355': 421700.0, '20230517180019': 419881.5, '20230519033308': 398153.0, '20230518075340':
    392645.0, '20230517153958': 383394.0, '20230508181757': 382659.5, '20230519213404': 381433.5, '20230427171131':
    371797.5, '20230510153843': 363580.0, '20230512211850': 346673.5, '20230517204451': 344752.0, '20230513095916':
    343119.0, '20230511224701': 325305.5, '20230523034033': 324572.0, '20230504125349': 305170.5, '20230512170431':
    283679.5, '20230520192625': 267580.0, '20230421192746': 250710.5, '20230525115626': 244668.0, '20230508220213':
    236812.0, '20230517025833': 235149.5, '20230510170242': 229573.5, '20230506111616': 228929.5, '20230518104908':
    220106.0, '20230522172834': 209487.5, '20230501040514': 206939.5, '20230519212155': 193494.5, '20230421215232':
    187840.5, '20230523020515': 182851.0, '20230512094635': 167667.0, '20230512105719': 166997.5, '20230517171727':
    161654.5, '20230512111225': 149758.5, '20230517104414': 137091.5, '20230505142626': 129104.0, '20230506145829':
    125879.5, '20230511201612': 121939.5, '20230512120728': 120640.0, '20230505164332': 116720.5, '20230517021606':
    113614.0, '20230518135715': 103445.5, '20230509163359': 101352.5, '20230505131816': 100129.5, '20230507125513':
    97431.0, '20230505135219': 95712.0, '20230503213852': 83442.0, '20230504094316': 82609.5, '20230517151648':
    73177.5, '20230517024455': 72298.0, '20230508080928': 71590.5, '20230513092954': 69709.5, '20230421204550':
    61537.0, '20230504223647': 59155.5, '20230506142341': 56811.0, '20230514173038': 50085.0, '20230506151750':
    43158.0, '20230511085916': 43130.0, '20230507175344': 42486.0, '20230506145035': 40613.0, '20230507172452':
    33549.0, '20230508164013': 29802.0, '20230507175928': 27326.5, '20230517164827': 26787.5, '20230524004853':
    20793.5, '20230503225234': 16082.5, '20230508131616': 7536.0, '20230506141535': 4687.5, '20230514182829': -1,
    '20230511150730': -1, '20230505150348': -1, '20230520105602': -1, '20230509144225': -1}"""
    base_data_dir = "./data"  # Raw unprocessed data from data_downloader.py
    train_val_dir = "./train_val_data"  # Data used for model training
    test_data_dir = "./eval_scrolls"  # Data used purely for testing, inference data source.
    orig_labels_dir = "./orig_labels"
    processed_labels_dir = "./labels"
    train_segment_ids = list(set([segment.split("_")[0] for segment in os.listdir("./labels") if
                                  segment.split("_")[0] in segment_id_data_paths]) - set(val_segment_ids))
    # train_segment_ids = ["20230530172803"]
    val_segment_ids = val_segment_ids
    # test_segment_ids = ['20230531101257']
    # 20231011144857 is the smallest one we'd care about.
    test_segment_ids = ['20230929220926',
                        '20231005123336',
                        '20231007101619',
                        '20231210121321',
                        '20231012184423',
                        '20231221180251',
                        '20231022170901',
                        '20231106155351',
                        '20231031143852',
                        '20230702185753',
                        '20231016151002']
    certain_no_ink_color = 40

    # ============== augmentation =============
    train_aug_list = [
        # A.ToFloat(max_value=65535.0),
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        # A.ToFloat(max_value=65535.0),
        A.RandomResizedCrop(
            size, size, scale=(0.85, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),
        # A.ChannelShuffle(p=0.5),
        A.OpticalDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0.0),
        A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                         normalized=True),
        A.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0.0),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.75
        ),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.9,
                           border_mode=cv2.BORDER_CONSTANT, value=0.0),
        A.OneOf([
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.PiecewiseAffine(p=0.5),  # IAAPiecewiseAffine
            A.MotionBlur(),
        ], p=0.9),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2),
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Resize(size, size),
        ChannelInvert(p=0.5),
        FourthAugment(p=1.0),
        A.Normalize(
            mean=[0] * in_channels,
            std=[1] * in_channels
        ),
        ToTensorV2(transpose_mask=True),
    ]
    # train_aug_list = [
    #     # A.ToFloat(max_value=65535.0),
    #     # A.RandomResizedCrop(
    #     #     size, size, scale=(0.85, 1.0)),
    #     FourthAugment(p=1.0),
    #     A.Resize(size, size),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     # A.RandomRotate90(p=0.6),
    #     # A.ChannelShuffle(p=0.5),
    #
    #     A.RandomBrightnessContrast(p=0.75),
    #     A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.75),
    #     A.OneOf([
    #         A.GaussNoise(var_limit=[10, 50]),
    #         A.GaussianBlur(),
    #         A.MotionBlur(),
    #     ], p=0.4),
    #     A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2),
    #                     mask_fill_value=0, p=0.5),
    #     # A.Cutout(max_h_size=int(size * 0.6),
    #     #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
    #     ChannelInvert(p=0.5),
    #     A.Normalize(
    #         mean=[0] * in_channels,
    #         std=[1] * in_channels
    #     ),
    #     ToTensorV2(transpose_mask=True),
    # ]

    valid_aug_list = [
        # A.ToFloat(max_value=65535.0),
        A.Resize(size, size),
        ChannelInvert(p=0.5),
        A.Normalize(
            mean=[0] * in_channels,
            std=[1] * in_channels
        ),
        ToTensorV2(transpose_mask=True),
    ]
