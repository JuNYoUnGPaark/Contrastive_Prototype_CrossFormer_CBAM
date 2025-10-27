#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================
# 0. 재현성
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# 1. 데이터 로드 & 전처리
# =========================================================
ACTIVITY_TO_ID = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5,
}
ID_TO_ACTIVITY = {v:k for k,v in ACTIVITY_TO_ID.items()}

def load_wisdm_raw(path_txt: str) -> pd.DataFrame:
    """
    WISDM_ar_v1.1_raw.txt 를 DataFrame으로 로드
    columns: user, activity, timestamp, x, y, z
    """
    cols = ["user","activity","timestamp","x","y","z"]
    df = pd.read_csv(path_txt, header=None, names=cols, on_bad_lines='skip')
    # z 컬럼에 붙은 ; 제거
    df["z"] = pd.to_numeric(df["z"].astype(str).str.replace(";","", regex=False),
                            errors='coerce')
    df.dropna(inplace=True)
    return df

def balance_by_oversampling(df: pd.DataFrame,
                            per_class_target: Dict[str,int]) -> pd.DataFrame:
    """
    간단 수동 오버샘플링/언더샘플링:
      - 각 activity별로 원하는 개수(per_class_target[act])만큼 행을 뽑는다.
      - 만약 원본이 적으면 반복 복제해서 늘린다.
      - 많으면 앞에서 자른다 (언더샘플링 느낌).
    결과적으로 클래스별 개수를 비슷하게 맞춘 balanced df 리턴.

    per_class_target 예:
      {'Walking':20000, 'Jogging':20000, 'Upstairs':20000,
       'Downstairs':20000, 'Sitting':20000, 'Standing':20000}
    """
    dfs = []
    for act, target_n in per_class_target.items():
        sub = df[df['activity']==act]
        if len(sub) == 0:
            continue
        if len(sub) >= target_n:
            dfs.append(sub.sample(n=target_n, random_state=42))
        else:
            # 부족하면 반복 replicate
            reps = target_n // len(sub) + 1
            sub_rep = pd.concat([sub]*reps, ignore_index=True)
            dfs.append(sub_rep.sample(n=target_n, random_state=42))
    out = pd.concat(dfs, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out

def make_sliding_windows(df: pd.DataFrame,
                         window_size=200,
                         step_size=20) -> Tuple[np.ndarray, np.ndarray]:
    """
    사용자/액티비티별로 구간을 끊고
    (x,y,z)를 window_size 길이로 슬라이싱, step_size stride로 밀어가며 윈도우를 만든다.
    X_out: (N, 3, T)
    y_out: (N,)
    """
    X_list = []
    y_list = []

    for (user, act), g in df.groupby(['user','activity']):
        sig = g[['x','y','z']].values.astype(np.float32)  # (L,3)
        label = ACTIVITY_TO_ID[act]

        L = len(sig)
        for start in range(0, L - window_size, step_size):
            chunk = sig[start:start+window_size]  # (T,3)
            X_list.append(chunk.T)               # (3,T)
            y_list.append(label)

    X_out = np.stack(X_list)                    # (N,3,T)
    y_out = np.array(y_list, dtype=np.int64)    # (N,)
    return X_out, y_out

# =========================================================
# 2. Dataset / Dataloader
# =========================================================
class SimpleWISDMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=False)

# =========================================================
# 3. 모델 (너 backbone 단일 헤드 버전)
#    여기선 contrast 끄고, 그냥 CE만 쓰는 버전으로 단순화
# =========================================================
class CBAMChannel(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c//reduction, c, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # x: (B,C,T)
        avg_out = self.fc(self.avg(x).squeeze(-1))
        max_out = self.fc(self.max(x).squeeze(-1))
        attn = self.sig((avg_out+max_out).unsqueeze(-1))
        return x * attn

class CBAMTemporal(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad = (k-1)//2
        self.conv = nn.Conv1d(2,1,kernel_size=k,padding=pad,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # x: (B,C,T)
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        a = torch.cat([avg_out,max_out], dim=1)    # (B,2,T)
        a = self.conv(a)                           # (B,1,T)
        a = self.sig(a)
        return x * a

class CBAM1D(nn.Module):
    def __init__(self, c, reduction=16, k=7):
        super().__init__()
        self.ca = CBAMChannel(c, reduction)
        self.ta = CBAMTemporal(k)
    def forward(self,x):
        x = self.ca(x)
        x = self.ta(x)
        return x

class CrossFormerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*2, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        # x: (B,T,C)
        h = self.norm1(x)
        attn_out,_ = self.attn(h,h,h)  # self-attn
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

class SimpleHARNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 seq_len=200,
                 embed_dim=64,
                 kernel_size=13,
                 dropout=0.1,
                 n_heads=8,
                 n_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim,
                      kernel_size=kernel_size,
                      padding=(kernel_size-1)//2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cbam = CBAM1D(embed_dim, reduction=8, k=kernel_size)

        self.xformer = CrossFormerBlock(
            dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, x):
        # x: (B,3,T)
        x = self.conv(x)         # (B,embed_dim,T)
        x = self.cbam(x)         # (B,embed_dim,T)

        x = x.transpose(1,2)     # (B,T,embed_dim)
        x = self.xformer(x)      # (B,T,embed_dim)
        x = x.transpose(1,2)     # (B,embed_dim,T)

        x = self.pool(x).squeeze(-1)  # (B,embed_dim)
        logits = self.cls(x)          # (B,n_classes)
        return logits

# =========================================================
# 4. 학습 / 평가 루프
# =========================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)

# =========================================================
# 5. main
# =========================================================
def main():
    config = {
        'DATA_PATH': 'C://Users/park9/CBAM_HAR/WISDM/WISDM_ar_v1.1_raw.txt',
        'SEED': 42,
        'TEST_SIZE': 0.2,     # random split ratio
        'VAL_SIZE': 0.2,      # from train portion
        'WINDOW_SIZE': 200,
        'STEP_SIZE': 100,
        'BATCH': 128,
        'EPOCHS': 80,
        'LR': 5e-4,
        'WEIGHT_DECAY': 1e-2,
        'EMBED_DIM': 64,
        'KERNEL_SIZE': 13,
        'N_HEADS': 8,
        'DROPOUT': 0.1,
    }

    seed_everything(config['SEED'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 1) 원본 로드
    df_raw = load_wisdm_raw(config['DATA_PATH'])

    # 2) 스케일링 (train 전에 fit 해야 하므로 잠깐 분할 필요)
    #    여기서는 아직 subject-wise 안 하니까 그냥 랜덤 train/test 먼저 자르고,
    #    나중에 train에서 fit한 scaler로 다시 전체 적용해도 돼.
    df_train_full, df_test = train_test_split(
        df_raw, test_size=config['TEST_SIZE'], shuffle=True, random_state=config['SEED']
    )
    df_train, df_val = train_test_split(
        df_train_full, test_size=config['VAL_SIZE'], shuffle=True, random_state=config['SEED']
    )

    # fit scaler on train only
    scaler = StandardScaler()
    scaler.fit(df_train[['x','y','z']])

    for d in [df_train, df_val, df_test]:
        d[['x','y','z']] = scaler.transform(d[['x','y','z']])

    # 3) 수동 오버샘플링으로 train 밸런스 맞추기
    #    전략: 각 클래스마다 동일한 타깃 수로 맞춘다.
    #    타깃 수는 train 안에서 가장 많은 클래스 count를 기준으로 골라도 되고,
    #    혹은 적당히 20000 이런 식으로 고정해도 된다.
    counts = df_train['activity'].value_counts().to_dict()
    max_count = max(counts.values())
    target_dict = {act: max_count for act in ACTIVITY_TO_ID.keys()}

    df_train_bal = balance_by_oversampling(df_train, target_dict)

    print("Class counts (before):", counts)
    print("Class counts (after):", df_train_bal['activity'].value_counts().to_dict())

    # 4) 윈도우 슬라이싱 (train_bal / val / test)
    X_train, y_train = make_sliding_windows(
        df_train_bal,
        window_size=config['WINDOW_SIZE'],
        step_size=config['STEP_SIZE']
    )
    X_val, y_val = make_sliding_windows(
        df_val,
        window_size=config['WINDOW_SIZE'],
        step_size=config['STEP_SIZE']
    )
    X_test, y_test = make_sliding_windows(
        df_test,
        window_size=config['WINDOW_SIZE'],
        step_size=config['STEP_SIZE']
    )

    print("Shapes:",
          X_train.shape, y_train.shape,
          X_val.shape, y_val.shape,
          X_test.shape, y_test.shape)

    train_ds = SimpleWISDMDataset(X_train, y_train)
    val_ds   = SimpleWISDMDataset(X_val,   y_val)
    test_ds  = SimpleWISDMDataset(X_test,  y_test)

    train_loader = make_loader(train_ds, config['BATCH'], shuffle=True)
    val_loader   = make_loader(val_ds,   config['BATCH'], shuffle=False)
    test_loader  = make_loader(test_ds,  config['BATCH'], shuffle=False)

    # 5) 모델 초기화
    model = SimpleHARNet(
        in_channels=3,
        seq_len=config['WINDOW_SIZE'],
        embed_dim=config['EMBED_DIM'],
        kernel_size=config['KERNEL_SIZE'],
        dropout=config['DROPOUT'],
        n_heads=config['N_HEADS'],
        n_classes=6
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['LR'],
        weight_decay=config['WEIGHT_DECAY']
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(config['EPOCHS']):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch+1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch+1) % 10 == 0:
            print(f"[{epoch+1:03d}/{config['EPOCHS']:03d}] "
                  f"TrainAcc={tr_acc:.4f} ValAcc={val_acc:.4f} ValF1={val_f1:.4f}")

    # 6) best로 테스트
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)

    print("\n================= RESULT =================")
    print(f"Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Test Acc    : {test_acc:.4f}")
    print(f"Test F1(w)  : {test_f1:.4f}")
    print("Classif Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=[ID_TO_ACTIVITY[i] for i in range(6)],
                                digits=4))

    cm = confusion_matrix(test_labels, test_preds, labels=[0,1,2,3,4,5])
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    ticks = [ID_TO_ACTIVITY[i] for i in range(6)]
    plt.xticks(range(6), ticks, rotation=45, ha='right')
    plt.yticks(range(6), ticks)
    for i in range(6):
        for j in range(6):
            plt.text(j,i,cm[i,j],ha='center',va='center',fontsize=8,color='black')
    plt.tight_layout()
    plt.savefig("cm_simple_balanced.png", dpi=200)
    plt.close()
    print("Saved cm_simple_balanced.png")

if __name__ == "__main__":
    main()

