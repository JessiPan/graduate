from torch.utils.data import DataLoader
from GeoSeg.geoseg.losses import *
from GeoSeg.geoseg.datasets.loveda_dataset import *
from GeoSeg.geoseg.models.UNetFormer import UNetFormer
from GeoSeg.tools.utils import Lookahead
from GeoSeg.tools.utils import process_model_params
from TorchCRF import CRF

# training hparam
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-CRF-r18-512crop-ms-epoch30-rep"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "last"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
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

def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=False,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# 定义后处理函数
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