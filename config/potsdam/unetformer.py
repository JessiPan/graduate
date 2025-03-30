
# 导入 PyTorch 相关模块
import torch
from torch.utils.data import DataLoader
# 导入 GeoSeg 库中的相关模块
from GeoSeg.geoseg.losses import *
from GeoSeg.geoseg.datasets.potsdam_dataset import *
from GeoSeg.geoseg.models.UNetFormer import UNetFormer
from GeoSeg.tools.utils import Lookahead
from GeoSeg.tools.utils import process_model_params

from GeoSeg.geoseg.datasets.potsdam_dataset import PotsdamDataset

# 定义训练的超参数
max_epoch = 45  # 最大训练轮数
ignore_index = len(CLASSES)  # 忽略的索引值，通常用于处理数据中的特殊类别
train_batch_size = 4  # 训练时的批量大小
val_batch_size = 4  # 验证时的批量大小
lr = 6e-4  # 学习率
weight_decay = 0.01  # 权重衰减系数
backbone_lr = 6e-5  # 骨干网络的学习率
backbone_weight_decay = 0.01  # 骨干网络的权重衰减系数
num_classes = len(CLASSES)  # 类别数量
classes = CLASSES  # 类别名称列表

# 定义模型权重、日志等相关的路径和参数
weights_name = "unetformer-CRF-r18-768crop-ms-e45"  # 模型权重的名称
weights_path = "model_weights/potsdam/{}".format(weights_name)  # 模型权重的保存路径
test_weights_name = "unetformer-CRF-r18-768crop-ms-e45"  # 测试时使用的模型权重名称
log_name = 'potsdam/{}'.format(weights_name)  # 日志的名称
monitor = 'val_F1'  # 监控的指标，用于模型保存等操作
monitor_mode = 'max'  # 监控指标的模式，这里是最大值
save_top_k = 1  # 保存最优的模型数量
save_last = True  # 是否保存最后一个模型
check_val_every_n_epoch = 1  # 每隔多少轮进行一次验证
pretrained_ckpt_path = None  # 预训练模型权重的路径，这里没有使用预训练模型
gpus = 'auto'  # 使用的 GPU 设置，这里设置为自动选择
resume_ckpt_path = None  # 是否从某个检查点继续训练，这里没有设置

# 定义网络模型
net = UNetFormer(num_classes=num_classes)  # 创建 UNetFormer 模型实例，指定类别数量

# 定义损失函数
loss = UnetFormerLoss(ignore_index=ignore_index)  # 创建 UnetFormerLoss 实例，指定忽略索引
use_aux_loss = True  # 是否使用辅助损失

# 定义数据加载器
train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)  # 创建训练数据集实例，指定数据根目录、模式、拼接比例和数据增强方式
val_dataset = PotsdamDataset(transform=val_aug)  # 创建验证数据集实例，指定数据增强方式
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)  # 创建测试数据集实例，指定数据根目录和数据增强方式

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=False,  # 是否将数据加载到 GPU 的 pin memory 中，这里设置为 False
                          shuffle=True,  # 是否打乱数据
                          drop_last=True)  # 是否丢弃最后一个不完整的批量

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# 定义优化器
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}  # 定义分层参数，用于设置骨干网络的学习率和权重衰减
net_params = process_model_params(net, layerwise_params=layerwise_params)  # 处理模型参数
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)  # 创建 AdamW 优化器实例
optimizer = Lookahead(base_optimizer)  # 使用 Lookahead 包装优化器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)  # 创建余弦退火重启学习率调度器实例

# 定义配置字典，用于存储一些配置项
config = {
    'data_root': '/mnt/c/Users/Jessi/PycharmProjects/airs/data/potsdam/train',  # 数据根目录
    'img_dir': 'images',  # 图像目录名称
    'mask_dir': 'masks',  # 掩码目录名称
    # 其他配置项
}