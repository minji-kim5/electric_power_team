# %%
import torch
print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
# %%
TRAIN_PATH = "train_rev9.csv"
TEST_PATH  = "test_rev.csv"
TIMECOL    = "측정일시"
TARGET     = "전기요금(원)"
ID_COL     = "id"

SEQ_LEN    = 96
BATCH_SIZE = 256
EPOCHS     = 40
LR         = 1e-3
VAL_RATIO  = 0.1
# %%
MONTH_WEIGHTS = {1:0.3, 2:0.3, 3:0.8, 4:0.8, 5:0.8, 6:1.0, 7:1.0, 8:1.0, 9:1.2, 10:1.5, 11:1.5, 12:1.0}
RECENCY_HALFLIFE_DAYS = 90

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
# %%
TRAIN_PATH = "train_rev9.csv"
TEST_PATH = "test_rev.csv"
# %%
season_map = {
            12: "겨울", 1: "겨울", 2: "겨울",
            3: "봄", 4: "봄", 5: "봄",
            6: "여름", 7: "여름", 8: "여름",
            9: "가을", 10: "가을", 11: "가을",
        }

# %%
def load_and_basic_prep(path):
    df = pd.read_csv(path, parse_dates=[TIMECOL])
    df["month"] = df[TIMECOL].dt.month
    df["hour"]  = df[TIMECOL].dt.hour
    df["minute"]= df[TIMECOL].dt.minute
    time_float  = df["hour"] + df["minute"]/60.0
    df["hour_sin"] = np.sin(2*np.pi*time_float/24.0)
    df["hour_cos"] = np.cos(2*np.pi*time_float/24.0)
    return df
# %%
train_df = load_and_basic_prep(TRAIN_PATH)
test_df  = load_and_basic_prep(TEST_PATH)
# %%
CAT_COLS = ["계절","작업유형","작업휴무"]
NUM_COLS = ["month","hour_sin","hour_cos"]
# %%
# def build_features(df, fit_cols=None):
#     num_cols = ["month", "hour_sin", "hour_cos"]
#     X_num = df[num_cols]
#     X_cat = pd.get_dummies(df[CAT_COLS], drop_first=False)
#     X = pd.concat([X_num, X_cat], axis=1)
#     if fit_cols is not None:
#         X = X.reindex(columns=fit_cols, fill_value=0)
#     return X

# X_train_full = build_features(train_df)
# fit_cols = X_train_full.columns
# X_test_full  = build_features(test_df, fit_cols=fit_cols)
# %%
train_cat = pd.get_dummies(train_df[CAT_COLS], drop_first=False)
test_cat  = pd.get_dummies(test_df[CAT_COLS],  drop_first=False).reindex(columns=train_cat.columns, fill_value=0)

X_train_df = pd.concat([train_df[NUM_COLS], train_cat], axis=1)
X_test_df  = pd.concat([test_df[NUM_COLS],  test_cat],  axis=1)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_df.values)
X_test_scaled  = scaler_X.transform(X_test_df.values)

if TARGET not in train_df.columns:
    raise ValueError("훈련 데이터에 타깃(전기요금(원)) 컬럼이 필요합니다.")
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(train_df[[TARGET]].values) 

X_exog_full = np.vstack([X_train_scaled[-SEQ_LEN:], X_test_scaled])
# %%
def make_supervised_sequences(X, y, seq_len=SEQ_LEN):
    """ X:(N,f), y:(N,1) -> (N-seq_len, seq_len, f), (N-seq_len, 1), seq_end_time_index """
    Xs, ys, idxs = [], [], []
    N = len(X)
    for i in range(N - seq_len):
        Xs.append(X[i:i+seq_len, :])
        ys.append(y[i+seq_len, 0])
        idxs.append(i+seq_len-1)
    return np.array(Xs), np.array(ys).reshape(-1,1), np.array(idxs)

X_seq, y_seq, idx_end = make_supervised_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
print("Train sequences:", X_seq.shape, y_seq.shape)
# %%
seq_end_time = pd.to_datetime(train_df.loc[idx_end, TIMECOL].to_numpy())

seq_month = pd.DatetimeIndex(seq_end_time).month.to_numpy()
month_w = np.array([MONTH_WEIGHTS.get(int(m), 1.0) for m in seq_month], dtype=float)

latest_ts = pd.to_datetime(train_df[TIMECOL].max())
days_from_end = ((latest_ts - seq_end_time) / np.timedelta64(1, 'D')).astype(float)

time_w = np.exp(-days_from_end / RECENCY_HALFLIFE_DAYS)

sample_w = np.asarray(month_w * time_w, dtype=float)
sample_w = sample_w / (sample_w.mean() + 1e-12)

# %%
class SeqDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)
        self.w = torch.tensor(w, dtype=torch.float32).view(-1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]

full_ds = SeqDataset(X_seq, y_seq, sample_w)

# 시간기반 검증(끝 10%) — 인덱스 순서 유지한 split
n_total = len(full_ds)
n_val   = max(1, int(n_total * VAL_RATIO))
n_train = n_total - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# %%
class GRURegressor(nn.Module):
    def __init__(self, n_feat, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size=n_feat, hidden_size=hidden1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: (B, T, F)
        o1, _ = self.gru1(x)           # (B,T,H1)
        o1 = self.drop(o1)
        o2, _ = self.gru2(o1)          # (B,T,H2)
        h = o2[:, -1, :]               # 마지막 타임스텝
        y = self.head(h).squeeze(-1)   # (B,)
        return y

n_feats = X_seq.shape[2]
model = GRURegressor(n_feat=n_feats).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# %%
def weighted_mae(pred, target, weight):
    # pred/target: (B,), weight: (B,)
    loss = torch.abs(pred - target)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
# %%
best_val = float("inf")
patience, bad = 8, 0

for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            pred = model(xb)
            loss = weighted_mae(pred, yb, wb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * len(xb)
    tr_loss /= n_train

    # validation
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb, wb in val_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            pred = model(xb)
            loss = weighted_mae(pred, yb, wb)  # 같은 MAE로 모니터
            va_loss += loss.item() * len(xb)
    va_loss /= n_val

    print(f"[{epoch:02d}/{EPOCHS}] train_mae={tr_loss:.6f}  val_mae={va_loss:.6f}")

    if va_loss + 1e-8 < best_val:
        best_val = va_loss
        bad = 0
        torch.save(model.state_dict(), "best_gru.pt")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break
# %%
model.load_state_dict(torch.load("best_gru.pt", map_location=DEVICE))
model.eval()
# %%
def predict_full_test(X_exog, seq_len=SEQ_LEN, batch=1024):
    N_test = X_exog.shape[0] - seq_len
    outs = []
    for start in range(0, N_test, batch):
        b = min(batch, N_test - start)
        win = []
        for j in range(b):
            i = start + j
            win.append(X_exog[i:i+seq_len, :])
        x = torch.tensor(np.stack(win, axis=0), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            y = model(x)
        outs.append(y.detach().cpu().numpy())
    pred_scaled = np.concatenate(outs, axis=0).reshape(-1)  # (len(test),)
    return pred_scaled

y_pred_scaled = predict_full_test(X_exog_full, SEQ_LEN)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
# %%
test_df_used = test_df.copy()  # 길이 동일
if len(test_df_used) != len(y_pred):
    L = min(len(test_df_used), len(y_pred))
    test_df_used = test_df_used.iloc[:L]
    y_pred = y_pred[:L]

submission = pd.DataFrame({
    ID_COL: test_df_used[ID_COL].values,
    "target": y_pred
})
submission.to_csv("submission_v2.csv", index=False, encoding="utf-8-sig")
# %%
print(submission.head())
# %%
submission.info()
# %%
