#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import copy
import random
import time
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from fvcore.nn import FlopCountAnalysis, parameter_count_table


# =================================================================================
# 0. 재현성 / 유틸
# =================================================================================
def seed_everything(seed=42):
    """
    재현성 확보: Python, NumPy, PyTorch 모두 시드 고정
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    DataLoader의 worker마다 난수 고정
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# In[ ]:


# -------------------------------------------
# 데이터 로드 / 기본 라벨 매핑
# -------------------------------------------
def load_full_dataframe(filepath: str) -> pd.DataFrame:
    """
    WISDM raw txt를 pandas DataFrame으로 로드.
    user, activity, timestamp, x, y, z
    z는 'xxxx;' 형태라 세미콜론 제거.
    """
    col_names = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(filepath, header=None, names=col_names, on_bad_lines='skip')
    df['z'] = pd.to_numeric(df['z'].astype(str).str.rstrip(';'), errors='coerce')
    df.dropna(axis=0, how='any', inplace=True)
    return df

class WISDMDataset(Dataset):
    """
    한 사용자/한 activity 구간씩 슬라이딩 윈도우를 뽑는다.
    우리는 4종 라벨을 다 저장해둔다:
      - y_full   : 최종 6-class {Walk,Jog,Up,Down,Sit,Stand} -> {0..5}
      - y_branch : branch(2-class) locomotion=0 / static=1
      - y_static : static 내부 2-class Sitting=0 / Standing=1 / else=-1
      - y_loco   : locomotion 내부 4-class Walk=0 Jog=1 Up=2 Down=3 / else=-1
    기본 __getitem__은 (x, y_full)를 반환하지만
    branch/static/loco 학습엔 아래 SubsetForTask 래퍼를 쓴다.
    """
    def __init__(self, dataframe, window_size=200, step_size=100):
        self.window_size = window_size
        self.step_size = step_size

        # 원래 6-class 인덱스 맵
        self.full_mapping = {
            'Walking': 0,
            'Jogging': 1,
            'Upstairs': 2,
            'Downstairs': 3,
            'Sitting': 4,
            'Standing': 5
        }

        def to_branch(a):
            # locomotion(걷기/뛰기/계단)=0, static(앉기/서기)=1
            if a in ['Walking','Jogging','Upstairs','Downstairs']:
                return 0
            else:
                return 1

        def to_static(a):
            # Sitting=0, Standing=1, else=-1 (해당없음)
            if a == 'Sitting':
                return 0
            elif a == 'Standing':
                return 1
            else:
                return -1

        def to_loco(a):
            # Walking=0, Jogging=1, Upstairs=2, Downstairs=3, else=-1
            if a == 'Walking':
                return 0
            elif a == 'Jogging':
                return 1
            elif a == 'Upstairs':
                return 2
            elif a == 'Downstairs':
                return 3
            else:
                return -1

        X_list = []
        y_full_list = []
        y_branch_list = []
        y_static_list = []
        y_loco_list = []

        #  subject별 / activity별로 끊어서 -> sliding window
        for (user, activity), group in dataframe.groupby(['user', 'activity']):
            # 슬라이딩 윈도우
            for i in range(0, len(group) - self.window_size, self.step_size):
                window = group.iloc[i:i+self.window_size]

                # 입력 시그널 (3, T)
                sig = window[['x','y','z']].values.T.astype(np.float32)
                X_list.append(sig)

                # 라벨들
                y_full_list.append(self.full_mapping[activity])
                y_branch_list.append(to_branch(activity))
                y_static_list.append(to_static(activity))
                y_loco_list.append(to_loco(activity))

        # numpy 배열화
        self.X = np.stack(X_list)  # (N, 3, window_size)
        self.y_full = np.array(y_full_list, dtype=np.int64)
        self.y_branch = np.array(y_branch_list, dtype=np.int64)
        self.y_static = np.array(y_static_list, dtype=np.int64)
        self.y_loco = np.array(y_loco_list, dtype=np.int64)

        print(f"[WISDMDataset] X={self.X.shape} full={self.y_full.shape} "
              f"branch={self.y_branch.shape} static={self.y_static.shape} "
              f"loco={self.y_loco.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 기본은 full 6-class 라벨을 반환 
        x = torch.from_numpy(self.X[idx])          # (3, window)
        y = torch.tensor(self.y_full[idx])         # scalar
        return x, y


# In[ ]:


class SubsetForTask(Dataset):
    """
    기존 WISDMDataset에서 특정 인덱스들만 골라 쓰고,
    어떤 라벨(y_branch / y_static / y_loco)을 target으로 쓸지 선택하는 래퍼.
    - base_dataset: WISDMDataset 인스턴스
    - indices: 사용할 샘플 인덱스 리스트/array
    - task: 'branch' | 'static' | 'loco'
    """
    def __init__(self, base_dataset, indices, task='branch'):
        self.base = base_dataset
        self.indices = np.array(indices)

        assert task in ['branch', 'static', 'loco'], "task must be one of ['branch','static','loco']"
        self.task = task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = torch.from_numpy(self.base.X[idx])  # (3, window)
        if self.task == 'branch':
            y = self.base.y_branch[idx]
        elif self.task == 'static':
            y = self.base.y_static[idx]
        elif self.task == 'loco':
            y = self.base.y_loco[idx]

        return x.float(), torch.tensor(int(y), dtype=torch.long)

# 추론 시 라벨 없이 X만 뽑고, 원래 인덱스를 기억하고 싶을 때:
class IndexOnlyXDataset(Dataset):
    def __init__(self, base_dataset: WISDMDataset, indices):
        self.base = base_dataset
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = torch.from_numpy(self.base.X[idx])  # (3,T)
        return x.float(), int(idx)


# In[ ]:


# =================================================================================
# 2. CBAM (1D 버전)
# =================================================================================
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, T)
        avg_out = self.avg_pool(x).squeeze(-1)  # (B, C)
        max_out = self.max_pool(x).squeeze(-1)  # (B, C)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = (avg_out + max_out).unsqueeze(-1)  # (B, C, 1)
        scale = self.sigmoid(out)
        return x * scale


class TemporalAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, T)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, T)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, T)

        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, T)
        out = self.conv(out)                        # (B, 1, T)
        out = self.sigmoid(out)
        return x * out


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.temporal_att = TemporalAttention1D(kernel_size)

    def forward(self, x):
        # x : (B, C, T)
        x = self.channel_att(x)
        x = self.temporal_att(x)
        return x


# In[ ]:


# =================================================================================
# 3. Contrastive Prototype Loss
# =================================================================================
class ContrastivePrototypeLoss(nn.Module):
    """
    각 클래스의 prototype과 feature를 InfoNCE 방식 loss
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        """
        Contrastive Loss between features and prototypes

        Args:
            features: (B, D) - 샘플 특징
            prototypes: (N_class, D) - 클래스별 프로토타입
            labels: (B,) - 레이블

        Returns:
            loss: contrastive loss
        """
        # L2 normalize
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # cosine similarity
        logits = torch.matmul(features, prototypes.t()) / self.temperature  # (B, num_classes)

        # InfoNCE Loss
        loss = F.cross_entropy(logits, labels)
        return loss


# In[ ]:


# =================================================================================
# 4. CrossFormer Block (Cross-Attn between tokens and learnable prototypes)
# =================================================================================
class ContrastCrossFormerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_prototypes=6,
                 n_heads=4,
                 mlp_ratio=2.0, 
                 dropout=0.1,
                 initial_prototypes=None):
        super().__init__()
        self.dim = dim
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))
        if initial_prototypes is not None:
            assert initial_prototypes.shape == self.prototypes.shape, \
                f"Shape mismatch: initial_prototypes {initial_prototypes.shape} vs self.prototypes {self.prototypes.shape}"
            self.prototypes.data.copy_(initial_prototypes)
            print(">>> [Main Model] Prototypes initialized with calculated mean features.")
        else:
            nn.init.xavier_uniform_(self.prototypes)
            print(">>> [Temporary Model or No Init Provided] Prototypes initialized with Xavier Uniform.")

        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
        self.proto_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x, return_proto_features=False, skip_cross_attention=False):
        B, T, C = x.shape
        attn_weights = None

        if not skip_cross_attention:
            normalized_prototypes = F.normalize(self.prototypes, dim=1)
            prototypes = normalized_prototypes.unsqueeze(0).expand(B, -1, -1)
            x_norm = self.norm1(x)
            cross_out, attn_weights = self.cross_attn(x_norm, prototypes, prototypes)
            x = x + cross_out

        x_norm = self.norm2(x)
        self_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self_out
        x = x + self.mlp(self.norm3(x))

        if return_proto_features:
            proto_features = x.mean(dim=1)
            proto_features = self.proto_proj(proto_features)
            return x, proto_features, attn_weights

        return x


# In[ ]:


# =================================================================================
# 5. 최종 HAR 모델: embedding + (CBAM) + CrossFormer + classifier
# =================================================================================
class ContrastCrossFormerCBAM_HAR(nn.Module):
    def __init__(self,
                 in_channels=9, 
                 seq_len=200,
                 embed_dim=64, 
                 reduced_dim=32,
                 n_classes=6, 
                 n_prototypes=6, 
                 n_heads=8,
                 kernel_size=7,
                 dropout=0.1,
                 temperature=0.07, 
                 initial_prototypes=None,
                 use_cbam=True,
                 use_crossformer=True,
                 use_contrast=True,
                 use_dim_reduction=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_crossformer = use_crossformer
        self.use_contrast = use_contrast
        self.use_dim_reduction = use_dim_reduction

        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(embed_dim), nn.GELU(), nn.Dropout(dropout)
        )

        if self.use_cbam:
            self.cbam = CBAM1D(embed_dim, reduction=8, kernel_size=kernel_size)

        working_dim = reduced_dim if use_dim_reduction else embed_dim
        if self.use_dim_reduction:
            self.dim_reduce = nn.Linear(embed_dim, reduced_dim)

        if self.use_crossformer:
            self.crossformer = ContrastCrossFormerBlock(
                dim=working_dim, n_prototypes=n_prototypes, n_heads=n_heads,
                mlp_ratio=2.0, dropout=dropout, initial_prototypes=initial_prototypes
            )
        else:
            self.self_attn = nn.TransformerEncoderLayer(
                d_model=working_dim, nhead=n_heads, dim_feedforward=int(working_dim * 2),
                dropout=dropout, batch_first=True
            )

        if self.use_dim_reduction:
            self.dim_restore = nn.Linear(reduced_dim, embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes)
        )

        if self.use_contrast and self.use_crossformer:
            self.contrast_loss = ContrastivePrototypeLoss(temperature=temperature)

    def forward(self, x, labels=None, return_contrast_loss=False):
        x = self.embedding(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = x.transpose(1, 2).contiguous()
        if self.use_dim_reduction:
            x = self.dim_reduce(x)

        proto_features = None
        if self.use_crossformer:
            if return_contrast_loss and self.use_contrast:
                x, proto_features, _ = self.crossformer(x, return_proto_features=True)
            else:
                x = self.crossformer(x, return_proto_features=False)
        else:
            x = self.self_attn(x)

        if self.use_dim_reduction:
            x = self.dim_restore(x)

        x = x.transpose(1, 2).contiguous()
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)

        if return_contrast_loss and self.use_contrast and proto_features is not None and labels is not None:
            contrast_loss = self.contrast_loss(proto_features, self.crossformer.prototypes, labels)
            return logits, contrast_loss

        return logits


# In[ ]:


# =================================================================================
# 6. 프로토타입 초기화: train data 평균 feature로 클래스별 prototype 만들기
# =================================================================================
def get_mean_prototypes(train_full_dataset, device, config):
    print("Calculating initial prototypes from mean features...")

    temp_model = ContrastCrossFormerCBAM_HAR(
        in_channels=config['in_channels'],
        seq_len=config['seq_len'],
        embed_dim=config['embed_dim'],
        reduced_dim=config['reduced_dim'], 
        n_heads=config['n_heads'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        use_cbam=True,
        use_crossformer=True, 
        use_contrast=False,
        use_dim_reduction=config['use_dim_reduction']
    ).to(device)

    temp_model.eval()

    temp_loader = DataLoader(train_full_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(temp_loader, desc="Prototype Init"):
            batch_x = batch_x.to(device)
            x = temp_model.embedding(batch_x)
            if temp_model.use_cbam:
                x = temp_model.cbam(x)
            x = x.transpose(1, 2).contiguous()
            if temp_model.use_dim_reduction:
                x = temp_model.dim_reduce(x)
            x = temp_model.crossformer(x, skip_cross_attention=True)
            x = x.transpose(1, 2).contiguous()
            pooled_features = temp_model.pool(x).squeeze(-1)
            all_features.append(pooled_features.cpu())
            all_labels.append(batch_y.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    working_dim = config['reduced_dim'] if config['use_dim_reduction'] else config['embed_dim']
    n_cls        = 6
    mean_proto   = torch.zeros(n_cls, working_dim)

    for c in range(n_cls):
        feats_c = all_features[all_labels == c]
        mean_proto[c] = feats_c.mean(dim=0) if len(feats_c)>0 else torch.randn(working_dim)

    return mean_proto.to(device)


# In[ ]:


# =================================================================================
# 7. 학습/평가 루프
# =================================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, use_contrast=True, contrast_weight=0.5):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_contrast_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in tqdm(dataloader, desc="train", leave=False):
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward
        if use_contrast and model.use_contrast and model.use_crossformer:
            logits, contrast_loss = model(batch_x, batch_y, return_contrast_loss=True)
            ce_loss = criterion(logits, batch_y)
            loss = ce_loss + contrast_weight * contrast_loss
            total_contrast_loss += contrast_loss.item()
        else:
            logits = model(batch_x)
            ce_loss = criterion(logits, batch_y)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    torch.cuda.synchronize() # 한 에폭 끝에서 동기화

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_contrast_loss = total_contrast_loss / len(dataloader) if total_contrast_loss > 0 else 0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, avg_ce_loss, avg_contrast_loss, acc, f1

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for xb, yb in dataloader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item())

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


# In[ ]:


# =================================================================================
# 8. Oversampling loader
# =================================================================================
def make_class_weights_for_dataset(dataset: SubsetForTask):
    ys = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        ys.append(int(y.item()))
    ys = np.array(ys)

    unique, counts = np.unique(ys, return_counts=True)
    invfreq = {c:1.0/counts[j] for j,c in enumerate(unique)}

    sample_weights = np.array([invfreq[int(label)] for label in ys], dtype=np.float32)
    return torch.from_numpy(sample_weights)

def make_loader(dataset, batch_size, oversample=True, shuffle=True):
    if oversample:
        sw = make_class_weights_for_dataset(dataset)
        sampler = WeightedRandomSampler(weights=sw,
                                        num_samples=len(sw),
                                        replacement=True)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            drop_last=False)
    else:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=False)
    return loader


# In[ ]:


# =================================================================================
# 9. 모델 생성/학습 헬퍼
# =================================================================================
def build_model_for_task(num_classes, config, device, initial_prototypes=None):
    model = ContrastCrossFormerCBAM_HAR(
        in_channels=config['in_channels'],
        seq_len=config['seq_len'],
        embed_dim=config['embed_dim'],
        reduced_dim=config['reduced_dim'],
        n_classes=num_classes,
        n_prototypes=num_classes,
        n_heads=config['n_heads'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        initial_prototypes=initial_prototypes,
        use_cbam=config['use_cbam'],
        use_crossformer=config['use_crossformer'],
        use_contrast=config['use_contrast'],
        use_dim_reduction=config['use_dim_reduction'],
    ).to(device)
    return model

def train_one_task(
    task_name,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    config,
    device
):
    """
    한 task(branch/static/loco)에 대해:
    - 모델 초기화
    - optimizer/scheduler
    - 에폭 반복하며 best ckpt 추적
    - best ckpt로 test 평가
    """
    model = build_model_for_task(
        num_classes=num_classes,
        config=config,
        device=device,
        initial_prototypes=None
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['EPOCHS']
    )

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(config['EPOCHS']):
        train_loss, train_ce, train_ctr, train_acc, train_f1 = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_contrast=config['use_contrast'],
            contrast_weight=config['contrast_weight']
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if (epoch+1) % 10 == 0:
            print(f"[{task_name}] Epoch {epoch+1:03d}/{config['EPOCHS']:03d} "
                  f"TrainLoss={train_loss:.4f} "
                  f"TrainAcc={train_acc:.4f} "
                  f"ValAcc={val_acc:.4f}")

    # best ckpt 로드 후 test
    assert best_state is not None, "No best state saved."
    model.load_state_dict(best_state)
    model.eval()

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"\n[{task_name}] Done!")
    print(f"  best val acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"  final test   : acc={test_acc:.4f} f1={test_f1:.4f}")

    return {
        'task': task_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'best_state': best_state,
    }

def init_trained_model(num_classes, config, device, state_dict):
    model = build_model_for_task(
        num_classes=num_classes,
        config=config,
        device=device,
        initial_prototypes=None
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

@torch.no_grad()
def predict_on_indices(model, base_dataset, indices, device, batch_size=256):
    """
    indices에 해당하는 window들만 추론.
    return: dict[idx] = pred_label(int)
    """
    ds = IndexOnlyXDataset(base_dataset, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = {}
    for xb, idx_original in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for i, raw_idx in enumerate(idx_original.numpy()):
            out[int(raw_idx)] = int(preds[i])
    return out

def majority_smooth(labels, k=5):
    """
    단순 majority smoothing (슬라이딩 윈도우 mode filter)
    labels: (N,)
    """
    labels = np.asarray(labels, dtype=int)
    N = len(labels)
    r = k // 2
    smoothed = np.empty_like(labels)
    for i in range(N):
        s = max(0, i-r)
        e = min(N, i+r+1)
        win = labels[s:e]
        binc = np.bincount(win)
        smoothed[i] = np.argmax(binc)
    return smoothed

def assemble_and_evaluate_final(
    test_dataset: WISDMDataset,
    branch_results,
    static_results,
    loco_results,
    config,
    device,
    smooth_k=5
):
    """
    2레벨 파이프라인으로 최종 6-class 예측:
    1) branch model이 전 윈도우를 static vs loco로 나눔
    2) static idx는 static model로 Sitting/Standing
       loco   idx는 loco   model로 Walk/Jog/Up/Down
    3) 합쳐서 최종 [0..5] 라벨로 복원
    4) smoothing 후 성능 보고
    """
    # 1. best state 로드
    branch_model = init_trained_model(
        num_classes=2,
        config=config,
        device=device,
        state_dict=branch_results['best_state']
    )
    static_model = init_trained_model(
        num_classes=2,
        config=config,
        device=device,
        state_dict=static_results['best_state']
    )
    loco_model = init_trained_model(
        num_classes=4,
        config=config,
        device=device,
        state_dict=loco_results['best_state']
    )

    N = len(test_dataset)
    all_idx = np.arange(N)

    # branch 예측: 0=locomotion, 1=static
    branch_pred_map = predict_on_indices(branch_model, test_dataset,
                                         all_idx, device,
                                         batch_size=config['BATCH_SIZE'])
    branch_pred_arr = np.array([branch_pred_map[i] for i in range(N)], dtype=int)

    pred_static_idx = np.where(branch_pred_arr == 1)[0]
    pred_loco_idx   = np.where(branch_pred_arr == 0)[0]

    # static 예측: 0 -> Sitting(4), 1 -> Standing(5)
    static_pred_map = predict_on_indices(static_model, test_dataset,
                                         pred_static_idx, device,
                                         batch_size=config['BATCH_SIZE'])
    # loco 예측:   0->Walking(0),1->Jogging(1),2->Upstairs(2),3->Downstairs(3)
    loco_pred_map   = predict_on_indices(loco_model, test_dataset,
                                         pred_loco_idx, device,
                                         batch_size=config['BATCH_SIZE'])

    final_pred = np.full(N, fill_value=-1, dtype=int)
    for idx in pred_static_idx:
        sp = static_pred_map[idx]  # 0 or 1
        final_pred[idx] = 4 if sp == 0 else 5
    for idx in pred_loco_idx:
        lp = loco_pred_map[idx]    # 0..3 already match 0..3
        final_pred[idx] = lp

    y_true = test_dataset.y_full.copy()

    final_pred_smooth = majority_smooth(final_pred, k=smooth_k)

    acc_raw = accuracy_score(y_true, final_pred)
    f1_raw  = f1_score(y_true, final_pred, average='weighted')
    acc_sm  = accuracy_score(y_true, final_pred_smooth)
    f1_sm   = f1_score(y_true, final_pred_smooth, average='weighted')

    target_names = ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']

    print("\n================ FINAL (6-class) ================")
    print(f"Raw     : Acc={acc_raw:.4f}, F1(w)={f1_raw:.4f}")
    print(f"Smoothed: Acc={acc_sm:.4f}, F1(w)={f1_sm:.4f}")

    print("\nClassification Report (Smoothed):")
    print(classification_report(y_true, final_pred_smooth,
                                target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, final_pred_smooth, labels=[0,1,2,3,4,5])
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    plt.xticks(range(6), target_names, rotation=45, ha='right')
    plt.yticks(range(6), target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Smoothed)')
    # annotate
    for i in range(6):
        for j in range(6):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig("confusion_matrix_final.png", dpi=200)
    plt.close()
    print("Saved confusion_matrix_final.png")

    return {
        'acc_raw': acc_raw,
        'f1_raw': f1_raw,
        'acc_smooth': acc_sm,
        'f1_smooth': f1_sm,
        'y_true': y_true,
        'pred_raw': final_pred,
        'pred_smooth': final_pred_smooth,
        'cm': cm,
    }


# In[ ]:


def main():
    config = {
        'DATA_DIR': 'C://Users/park9/CBAM_HAR/WISDM',
        'BATCH_SIZE': 256,
        'EPOCHS': 100,
        'SEED': 42,
        'LEARNING_RATE': 5e-4,
        'WEIGHT_DECAY': 1e-2,

        'in_channels': 3,
        'seq_len': 200,
        'step_size': 100,

        'embed_dim': 64,
        'reduced_dim': 32,
        'n_heads': 8,
        'kernel_size': 13,
        'dropout': 0.1,

        'use_cbam': True,
        'use_crossformer': True,
        'use_contrast': True,
        'use_dim_reduction': False,

        'temperature': 0.05,
        'contrast_weight': 0.1,
    }

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(config['SEED'])
    print(f"Device: {DEVICE}")
    print(f"Loading WISDM Dataset from: {config['DATA_DIR']}")

    # 1. raw 로드
    full_df = load_full_dataframe(
        os.path.join(config['DATA_DIR'], 'WISDM_ar_v1.1_raw.txt')
    )

    # 2. subject-wise split
    all_users = full_df['user'].unique()
    train_val_users, test_users = train_test_split(
        all_users, test_size=0.2, random_state=config['SEED']
    )
    train_users, val_users = train_test_split(
        train_val_users, test_size=0.2, random_state=config['SEED']
    )

    train_df = full_df[full_df['user'].isin(train_users)].copy()
    val_df   = full_df[full_df['user'].isin(val_users)].copy()
    test_df  = full_df[full_df['user'].isin(test_users)].copy()
    print("Users per split:",
          len(train_users), len(val_users), len(test_users))

    # 3. scaling
    scaler = StandardScaler()
    scaler.fit(train_df[['x','y','z']])
    for df_ in [train_df, val_df, test_df]:
        df_[['x','y','z']] = scaler.transform(df_[['x','y','z']])

    # 4. window → base dataset
    train_dataset = WISDMDataset(train_df,
                                 window_size=config['seq_len'],
                                 step_size=config['step_size'])
    val_dataset   = WISDMDataset(val_df,
                                 window_size=config['seq_len'],
                                 step_size=config['step_size'])
    test_dataset  = WISDMDataset(test_df,
                                 window_size=config['seq_len'],
                                 step_size=config['step_size'])
    print(f"Windows per split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # 5. 인덱스 분리
    train_idx_branch = np.arange(len(train_dataset))
    val_idx_branch   = np.arange(len(val_dataset))
    test_idx_branch  = np.arange(len(test_dataset))

    train_idx_static = np.where(train_dataset.y_static != -1)[0]
    val_idx_static   = np.where(val_dataset.y_static != -1)[0]
    test_idx_static  = np.where(test_dataset.y_static != -1)[0]

    train_idx_loco = np.where(train_dataset.y_loco != -1)[0]
    val_idx_loco   = np.where(val_dataset.y_loco != -1)[0]
    test_idx_loco  = np.where(test_dataset.y_loco != -1)[0]

    print(f"[branch] train/val/test = {len(train_idx_branch)}, {len(val_idx_branch)}, {len(test_idx_branch)}")
    print(f"[static] train/val/test = {len(train_idx_static)}, {len(val_idx_static)}, {len(test_idx_static)}")
    print(f"[loco]   train/val/test = {len(train_idx_loco)}, {len(val_idx_loco)}, {len(test_idx_loco)}")

    # 6. SubsetForTask
    train_branch_ds = SubsetForTask(train_dataset, train_idx_branch, task='branch')
    val_branch_ds   = SubsetForTask(val_dataset,   val_idx_branch,   task='branch')
    test_branch_ds  = SubsetForTask(test_dataset,  test_idx_branch,  task='branch')

    train_static_ds = SubsetForTask(train_dataset, train_idx_static, task='static')
    val_static_ds   = SubsetForTask(val_dataset,   val_idx_static,   task='static')
    test_static_ds  = SubsetForTask(test_dataset,  test_idx_static,  task='static')

    train_loco_ds   = SubsetForTask(train_dataset, train_idx_loco,   task='loco')
    val_loco_ds     = SubsetForTask(val_dataset,   val_idx_loco,     task='loco')
    test_loco_ds    = SubsetForTask(test_dataset,  test_idx_loco,    task='loco')

    print(f"[Subset sizes] branch_train={len(train_branch_ds)}, static_train={len(train_static_ds)}, loco_train={len(train_loco_ds)}")

    # 7. DataLoaders
    BATCH = config['BATCH_SIZE']
    train_branch_loader = make_loader(train_branch_ds, BATCH, oversample=True)
    val_branch_loader   = make_loader(val_branch_ds,   BATCH, oversample=False, shuffle=False)
    test_branch_loader  = make_loader(test_branch_ds,  BATCH, oversample=False, shuffle=False)

    train_static_loader = make_loader(train_static_ds, BATCH, oversample=True)
    val_static_loader   = make_loader(val_static_ds,   BATCH, oversample=False, shuffle=False)
    test_static_loader  = make_loader(test_static_ds,  BATCH, oversample=False, shuffle=False)

    train_loco_loader   = make_loader(train_loco_ds,   BATCH, oversample=True)
    val_loco_loader     = make_loader(val_loco_ds,     BATCH, oversample=False, shuffle=False)
    test_loco_loader    = make_loader(test_loco_ds,    BATCH, oversample=False, shuffle=False)

    print("✔ Dataloaders prepared (branch/static/loco).")

    # 8. task별 학습
    branch_results = train_one_task(
        task_name="branch",
        num_classes=2,
        train_loader=train_branch_loader,
        val_loader=val_branch_loader,
        test_loader=test_branch_loader,
        config=config,
        device=DEVICE
    )

    static_results = train_one_task(
        task_name="static",
        num_classes=2,
        train_loader=train_static_loader,
        val_loader=val_static_loader,
        test_loader=test_static_loader,
        config=config,
        device=DEVICE
    )

    loco_results = train_one_task(
        task_name="loco",
        num_classes=4,
        train_loader=train_loco_loader,
        val_loader=val_loco_loader,
        test_loader=test_loco_loader,
        config=config,
        device=DEVICE
    )

    print("\nAll three tasks trained!")
    print("branch:", branch_results['test_acc'], branch_results['test_f1'])
    print("static:", static_results['test_acc'], static_results['test_f1'])
    print("loco  :", loco_results['test_acc'],   loco_results['test_f1'])

    # 9. 최종 6-class 조립 + smoothing 평가
    final_pack = assemble_and_evaluate_final(
        test_dataset=test_dataset,
        branch_results=branch_results,
        static_results=static_results,
        loco_results=loco_results,
        config=config,
        device=DEVICE,
        smooth_k=5
    )

    print("\nFinal (6-class) Weighted F1 after smoothing:",
            f"{final_pack['f1_smooth']:.4f}")

    print("Done.")

if __name__ == "__main__":
    main()

