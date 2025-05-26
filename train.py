import os

from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm as tqdm

from torchvision.models import efficientnet_v2_m, efficientnet_v2_s
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from warmup_scheduler import GradualWarmupScheduler
import albumentations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import albumentations as A
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings('ignore')

# make dirs
os.makedirs("models", exist_ok=True)
os.makedirs("log", exist_ok=True)

# Fix SEED
# Fix SEED
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

data_dir = './'

DEBUG = False
FOLD = 1

kernel_type = 'efficientnet-m-{}'.format(FOLD)
modelname = 'efficientnet-m'

tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 2
num_workers = 6
out_dim = 5
init_lr = 1e-4
warmup_factor = 10
n_epochs = 20
patience = 5

load_raw_png = False
load_jpg = False

mixup = True
cutmix = False

# Cosine annealing or exp scheduler
COSINE = True
EXP = False
if not COSINE:
    EXP = True

poolmethod = "avgpool"

kernel_type += "_tile{}_imsize{}".format(n_tiles, image_size)

load_checkpoint = False
checkpoint_id = 19


# load clean labels
df_train = pd.read_csv(os.path.join(data_dir, "data_csv/train_9k.csv"))
# df_val = pd.read_csv(os.path.join(data_dir,"data_csv/test_9k.csv"))
image_folder = os.path.join(data_dir, 'train_256_36')

warmup_epo = 1
device = torch.device('cuda')

print(image_folder)
print(df_train.head())

skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1

splitted = skf.split(df_train, df_train['isup_grade'])
for i, (train_idx, valid_idx) in enumerate(splitted):
    df_train.loc[valid_idx, 'fold'] = i
print(df_train.head())
print(df_train.tail())

train_idx = np.where((df_train['fold'] != FOLD))[0]
valid_idx = np.where((df_train['fold'] == FOLD))[0]

df_this = df_train.loc[train_idx].reset_index(drop=True)
df_valid = df_train.loc[valid_idx].reset_index(drop=True)


def erase(df_train):
    df_train2 = df_train
    erase = []
    for i, id in enumerate(df_train2["image_id"].to_numpy()):
        if not os.path.isfile(os.path.join(data_dir, f'{id}.npz')):
            erase.append(i)
            pass
    return df_train.drop(erase)


df_valid = erase(df_valid).reset_index()


class enet(nn.Module):
    def __init__(self, out_dim=5):
        super(enet, self).__init__()
        self.basemodel = EfficientNet.from_pretrained(modelname)
        self.myfc = nn.Linear(self.basemodel._fc.in_features, out_dim)
        self.basemodel._fc = nn.Identity()

    def extract(self, x):
        return self.basemodel(x)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.myfc(x)
        return x


class enetv2(nn.Module):
    def __init__(self, out_dim=5, types="s", freeze_id=3):
        super(enetv2, self).__init__()

        if types == "s":
            self.basemodel = efficientnet_v2_s(weights="DEFAULT")
        else:
            self.basemodel = efficientnet_v2_m(weights="DEFAULT")

        in_features = self.basemodel.classifier[1].in_features

        self.basemodel.classifier[1] = nn.Identity()

        self.myfc = nn.Linear(in_features, out_dim)

        self.freeze_first_n_blocks(freeze_id)

    def freeze_first_n_blocks(self, n=3):
        """
        Freezes the first `n` blocks of the EfficientNetV2 model.
        """
        blocks = self.basemodel.features

        err_msg = f"too many blocks, model has only {len(blocks)} blocks"
        assert n <= len(blocks), err_msg

        for i in range(n):
            for param in blocks[i].parameters():
                param.requires_grad = False

    def extract(self, x):
        return self.basemodel(x)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.myfc(x)
        return x


def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_tiles(img, mode=0, transform=None):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,
                  [[pad_h // 2, pad_h - pad_h // 2],
                   [pad_w // 2, pad_w - pad_w//2],
                   [0, 0]], constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    ).astype(np.float32)

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1)
                         < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles-len(img3)],
                             [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    img3 = (img3*255).astype(np.uint8)
    for i in range(len(img3)):
        if transform is not None:
            img3[i] = transform(image=img3[i])['image']
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                 val=False
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform
        self.val = val

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        if load_raw_png:
            tiff_file = os.path.join(image_folder, f'{img_id}.jpg')
            image = cv2.imread(tiff_file)
            if self.transform is not None:
                tiles, OK = get_tiles(image, self.tile_mode, self.transform)
            else:
                tiles, OK = get_tiles(image, self.tile_mode)

            if self.rand:
                idxes = np.random.choice(list(range(self.n_tiles)),
                                         self.n_tiles,
                                         replace=False)
            else:
                idxes = list(range(self.n_tiles))

            n_row_tiles = int(np.sqrt(self.n_tiles))
            images = np.zeros((image_size * n_row_tiles,
                               image_size * n_row_tiles,
                               3))

            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    i = h * n_row_tiles + w

                    if len(tiles) > idxes[i]:
                        this_img = tiles[idxes[i]]['img']
                    else:
                        this_img = np.ones((self.image_size,
                                            self.image_size,
                                            3)).astype(np.uint8) * 255

                    this_img = 255 - this_img
                    if self.transform is not None:
                        this_img = self.transform(image=this_img)['image']
                    h1 = h * image_size
                    w1 = w * image_size
                    images[h1:h1+image_size, w1:w1+image_size] = this_img
                    images = (images*255).astype(np.uint8)
        elif load_jpg:
            file = os.path.join("train_{}_{}_aug".format(self.image_size,
                                                         self.n_tiles),
                                f'{img_id}_{np.random.randint(0, 9)}.jpg')
            images = cv2.imread(file)
        else:
            file = os.path.join(image_folder, f'{img_id}.npz')
            images = np.load(file)["arr_0"]

        # Load labels
        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        n_row_tiles = int(np.sqrt(self.n_tiles))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                h1 = h * image_size
                w1 = w * image_size
                this_img = images[h1:h1+image_size, w1:w1+image_size]
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']

                images[h1:h1+image_size, w1:w1+image_size] = this_img
        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']

        # # Mixup part
        rd = np.random.rand()
        if mixup and rd < 0.3 and self.transform is not None:
            mix_idx = np.random.randint(0, len(self.df))
            row2 = self.df.iloc[mix_idx]
            img_id2 = row2.image_id
            file = os.path.join(image_folder, f'{img_id2}.npz')
            images2 = np.load(file)["arr_0"]
            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    h1 = h * image_size
                    w1 = w * image_size
                    this_img = images2[h1:h1+image_size, w1:w1+image_size]
                    if self.transform is not None:
                        this_img = self.transform(image=this_img)['image']

                images2[h1:h1+image_size, w1:w1+image_size] = this_img

            if self.transform is not None:
                images2 = self.transform(image=images2)['image']

            # blend image
            gamma = np.random.beta(1, 1)
            images = ((images*gamma + images2*(1-gamma))).astype(np.uint8)
            # blend labels
            label2 = np.zeros(5).astype(np.float32)
            label2[:row2.isup_grade] = 1.
            label = (label*gamma+label2*(1-gamma))

        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        return torch.tensor(images), torch.tensor(label)


dataset_show = PANDADataset(df_train, image_size, n_tiles, 0, transform=None)
images, label = dataset_show[0]

transforms_train = A.Compose([
    A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=180, p=0.5),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=0.2,
                             sat_shift_limit=0.2,
                             val_shift_limit=0.2,
                             p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   p=0.5),
    ], p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 10),
        hole_height_range=(10, 128),
        hole_width_range=(10, 128),
        ill=0, p=0.5),
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5)
])

transforms_val = albumentations.Compose([])

criterion = nn.BCEWithLogitsLoss()

scaler = GradScaler()


def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for idx, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()

        with autocast():
            logits = model(data)
            loss = loss_func(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        lr_print = f"lr: {optimizer.param_groups[0]['lr']: .2e}"
        smooth_print = f"smth: {smooth_loss: .4f}"
        loss_print = f"loss: {loss_np: .4f}"

        bar.set_postfix_str(f"{loss_print}, {smooth_print}, {lr_print}")
    return train_loss


def val_epoch(loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    bar = tqdm(loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(bar):
            data, target = data.to(device), target.to(device)
            with autocast():
                logits = model(data)

                loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))

            val_loss.append(loss.detach().cpu().numpy())
            bar.set_postfix_str(f"val_loss: {np.mean(val_loss): .4f}")
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.

    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(
        PREDS[df_valid['data_provider'] == 'karolinska'],
        df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values,
        weights='quadratic')
    qwk_r = cohen_kappa_score(
        PREDS[df_valid['data_provider'] == 'radboud'],
        df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values,
        weights='quadratic')

    if EXP:
        scheduler.step(val_loss)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk, qwk_k, qwk_r, PREDS, TARGETS


dataset_train = PANDADataset(df_this, image_size,
                             n_tiles, transform=transforms_train)
dataset_valid = PANDADataset(df_valid, image_size,
                             n_tiles, transform=None, val=True)

# Setup dataloader
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    sampler=RandomSampler(dataset_train), num_workers=num_workers,
    pin_memory=True, persistent_workers=True
    )

valid_loader = torch.utils.data.DataLoader(
    dataset_valid, batch_size=batch_size,
    sampler=SequentialSampler(dataset_valid), num_workers=num_workers,
    pin_memory=True, persistent_workers=True
    )

# Initialize model
if 'b0' in modelname or 'b1' in modelname:
    model = enet(out_dim=out_dim)
elif '-m' in modelname:
    model = enetv2(out_dim=out_dim, types='m')
elif '-s' in modelname:
    model = enetv2(out_dim=out_dim, types='s')
model = model.to(device)

len_t = len(dataset_train)
len_v = len(dataset_valid)
print(f"[INFO] Train samples: {len_t} - Valid samples: {len_v}")

# We use Cosine annealing LR scheduling
if COSINE:
    optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        n_epochs-warmup_epo)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=warmup_factor,
        total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
else:
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                               patience=4, verbose=True,
                                               min_lr=1e-3*1e-5, factor=0.5)


os.makedirs("models", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs(os.path.join('models', kernel_type), exist_ok=True)
gmt7_timezone = pytz.timezone('Asia/Bangkok')
best_file = f'{kernel_type}_best_model.pth'
last_file = f'{kernel_type}_last_model.pth'

if load_checkpoint:
    checkpoint = torch.load(
        os.path.join(os.path.join('models', kernel_type),
                     f'{kernel_type}_model_{checkpoint_id}.pth'),
        map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch'] + 1

    qwk_max = checkpoint['qwk']
    print(f"Current best qwk: {qwk_max: .5f}")
else:
    epoch_start = 0
    qwk_max = 0.0

threshold = 0
for epoch in range(epoch_start, epoch_start + n_epochs):
    torch.cuda.empty_cache()
    print(datetime.now(gmt7_timezone).strftime('%Y-%m-%d %H:%M:%S'),
          'Epoch:', epoch+1)
    if COSINE:
        scheduler.step(epoch)
    train_loss = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk, qwk_k, qwk_r, TARGETS, PREDS = val_epoch(valid_loader)

    ep_p = f"Epoch {epoch+1: 2d}"
    lr_p = f"lr: {optimizer.param_groups[0]["lr"]: .2e}"
    tl_p = f"train loss: {np.mean(train_loss): .5f}"
    vl_p = f"val loss: {np.mean(val_loss): .5f}"
    acc_p = f"acc: {(acc): .5f}"
    qwk_p = f"qwk: {(qwk): .5f}, qwk_k: {(qwk_k): .5f}, qwk_r: {(qwk_r): .5f}"
    content = f'{ep_p}, {lr_p}, {tl_p}, {vl_p}, {acc_p},     {qwk_p}'

    print(datetime.now(
        gmt7_timezone).strftime('%Y-%m-%d %H:%M:%S') + ' ' + content)
    with open(f'log/log_{kernel_type}.txt', 'a') as appender:
        appender.write(content + '\n')

    cmat = confusion_matrix(TARGETS, PREDS).astype("uint")
    with open(f'log/logcmat_{kernel_type}.txt', 'a') as appender:
        appender.write("Epoch: " + str(epoch+1) + '\n')
        np.savetxt(appender, cmat, fmt='%3d')

    # Save all model
    checkpoint = {
        'epoch': epoch,
        'qwk': qwk,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint,
               os.path.join(os.path.join('models', kernel_type),
                            f'{kernel_type}_model_{epoch}.pth'))

    if epoch < 10:
        threshold = 0

    if qwk > qwk_max:
        print(f'Best qwk ({qwk_max: .5f} --> {qwk: .5f}).  Saving best ...')
        qwk_max = qwk
        threshold = 0
    else:
        threshold += 1

    if threshold > patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
