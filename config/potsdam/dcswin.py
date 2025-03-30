from torch.utils.data import DataLoader
from GeoSeg.geoseg.losses import *
from GeoSeg.geoseg.datasets.potsdam_dataset import *
from GeoSeg.geoseg.models.DCSwin import dcswin_small
from GeoSeg.tools.utils import Lookahead
from GeoSeg.tools.utils import process_model_params
from timm.scheduler.poly_lr import PolyLRScheduler

# 定义训练超参数
max_epoch = 30  # 最大训练轮数
ignore_index = len(CLASSES)  # 在损失计算中忽略的类别索引
train_batch_size = 8  # 训练时的批量大小
val_batch_size = 4  # 验证时的批量大小
lr = 1e-3  # 学习率
weight_decay = 2.5e-4  # 权重衰减
backbone_lr = 1e-4  # 主干网络的学习率
backbone_weight_decay = 2.5e-4  # 主干网络的权重衰减
num_classes = len(CLASSES)  # 类别数量
classes = CLASSES  # 类别名称

# 定义模型权重和日志的路径
weights_name = "dcswin-small-1024-ms-512crop-e30"  # 模型权重文件名
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "dcswin-small-1024-ms-512crop-e30"
log_name = 'potsdam/{}'.format(weights_name)  # 日志文件名
monitor = 'val_F1'  # 监控的指标（用于保存最佳模型）
monitor_mode = 'max'  # 监控指标的模式（最大值或最小值）
save_top_k = 1  # 保存最好的 k 个模型权重
save_last = False  # 是否保存最后一个模型权重
check_val_every_n_epoch = 1  # 每多少个 epoch 进行一次验证
pretrained_ckpt_path = None  # 预训练模型权重路径
gpus = 'auto'  # 使用的 GPU 配置
resume_ckpt_path = None  # 恢复训练时的检查点路径

# 定义网络模型
net = dcswin_small(num_classes=num_classes)  # 初始化 DCSwin 模型

# 定义损失函数
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),  # 平滑交叉熵损失
    DiceLoss(smooth=0.05, ignore_index=ignore_index),  # Dice 损失
    1.0, 1.0  # 两种损失的权重
)
use_aux_loss = False  # 是否使用辅助损失

# 定义数据增强和数据加载器
def get_training_transform(albu=None):
    """定义训练时的数据增强"""
    train_transform = [
        albu.RandomRotate90(p=0.5),  # 随机旋转 90 度
        albu.Normalize()  # 归一化
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    """应用训练时的数据增强"""
    crop_aug = Compose([
        RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),  # 随机缩放
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)  # 智能裁剪
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def get_val_transform(albu=None):
    """定义验证时的数据增强"""
    val_transform = [
        albu.Normalize()  # 归一化
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    """应用验证时的数据增强"""
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

# 创建训练、验证和测试数据集
train_dataset = PotsdamDataset(
    data_root='data/potsdam/train',  # 训练数据路径
    mode='train',  # 模式为训练
    mosaic_ratio=0.25,  # 马赛克数据增强的比例
    transform=train_aug  # 训练时的数据增强
)

val_dataset = PotsdamDataset(
    transform=val_aug  # 验证时的数据增强
)

test_dataset = PotsdamDataset(
    data_root='data/potsdam/test',  # 测试数据路径
    transform=val_aug  # 测试时的数据增强
)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_dataset,  # 训练数据集
    batch_size=train_batch_size,  # 训练时的批量大小
    num_workers=4,  # 数据加载的线程数
    pin_memory=True,  # 是否将数据加载到 GPU 的内存中
    shuffle=True,  # 是否打乱数据
    drop_last=True  # 是否丢弃最后一个不完整的批次
)

val_loader = DataLoader(
    dataset=val_dataset,  # 验证数据集
    batch_size=val_batch_size,  # 验证时的批量大小
    num_workers=4,  # 数据加载的线程数
    shuffle=False,  # 是否打乱数据
    pin_memory=True,  # 是否将数据加载到 GPU 的内存中
    drop_last=False  # 是否丢弃最后一个不完整的批次
)

# 定义优化器
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}  # 主干网络的参数
net_params = process_model_params(net, layerwise_params=layerwise_params)  # 处理模型参数
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)  # 使用 AdamW 优化器
optimizer = Lookahead(base_optimizer)  # 使用 Lookahead 优化器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # 使用余弦退火学习率调度器