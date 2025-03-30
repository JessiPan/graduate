import numpy as np
from TorchCRF import CRF  # 修正导入语句
from torch.utils.data import DataLoader
from GeoSeg.geoseg.losses import *
from GeoSeg.geoseg.datasets.vaihingen_dataset import *
from GeoSeg.geoseg.models.UNetFormer import UNetFormer
from GeoSeg.tools.utils import Lookahead
from GeoSeg.tools.utils import process_model_params

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-CRF-r18-512-crop-ms-e105"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "unetformer-CRF-r18-512-crop-ms-e105"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = UNetFormer(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

def crf_post_processing(logits, images):
    """
    使用 CRF 进行后处理
    :param logits: 模型输出的 logits，形状为 (batch_size, num_classes, height, width)
    :param images: 输入图像，形状为 (batch_size, channels, height, width)
    :return: 经过 CRF 优化后的预测结果，形状为 (batch_size, height, width)
    """
    batch_size, num_classes, height, width = logits.shape
    logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # (batch_size * height * width, num_classes)
    images = images.permute(0, 2, 3, 1).contiguous().view(-1, images.shape[1])  # (batch_size * height * width, channels)

    # 将 logits 转换为概率
    probs = torch.softmax(logits, dim=1)

    # 将图像和概率传递给 CRF
    crf_emissions = probs.view(batch_size, height, width, num_classes)
    crf_images = images.view(batch_size, height, width, -1)

    # 初始化 CRF
    crf = CRF(num_tags=num_classes, batch_first=True)  # 修正 CRF 实例化

    # 应用 CRF
    crf_logits = crf(crf_emissions, crf_images)

    # 将 CRF 的输出转换为预测结果
    crf_preds = torch.argmax(crf_logits, dim=3)  # (batch_size, height, width)

    return crf_preds

# 测试时应用 CRF
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            logits = model(images)  # 获取模型的输出
            crf_preds = crf_post_processing(logits, images)  # 应用 CRF 后处理
            # 这里可以将 crf_preds 保存或进一步处理