import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import glob
import os
from tqdm import tqdm
import math
from scipy.signal import find_peaks
# ==============================================================================
# 1. 설정 및 하이퍼파라미터
# ==============================================================================
CONFIG = {
    "data_path": "processed_dataset",
    "save_path": "checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "max_seq_len": 512, # 모델이 한 번에 처리할 시퀀스(프레임) 길이
    "checkpoint_path": "checkpoints/latest_checkpoint.pth",
    # 모델 파라미터
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,
}

# 재현성을 위한 시드 고정
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs(CONFIG["save_path"], exist_ok=True)

# ==============================================================================
# 2. 모델 정의 (이전 코드와 동일, Sigmoid만 제거)
# ==============================================================================
class PositionalEncoding(nn.Module):
    # ... (이전과 동일한 PositionalEncoding 코드) ...
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :]

class RhythmFormer(nn.Module):
    def __init__(self, input_channels=3, d_model=256, nhead=8, difficulty_dim=11, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(RhythmFormer, self).__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        )
        self.feature_projection = nn.Linear(128 * 3 * 20 + difficulty_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 4 * 2)

    def forward(self, x, difficulty_vector):
        # x shape: (batch_size, seq_len, C, H, W) e.g., (16, 512, 3, 15, 80)
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        difficulty_vector = difficulty_vector.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([x, difficulty_vector], dim=-1)
        x = self.feature_projection(combined_features)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = self.output_layer(x)
        # BCEWithLogitsLoss를 사용하므로 마지막 Sigmoid는 제거
        return x.view(batch_size, seq_len, 4, 2)

# ==============================================================================
# 3. 데이터셋 클래스 및 DataLoader 정의
# ==============================================================================
class OsuDataset(Dataset):
    def __init__(self, file_paths, max_seq_len):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        # TODO: 난이도 벡터(od, density)를 모델 입력에 추가하는 로직 (다음 단계)
        x, y = data['x'], data['y']
        od = data['od']
        density = data['density']
        difficulty_vector = np.concatenate([od, density],axis = 0)

        # 데이터를 max_seq_len 길이로 자르거나 패딩
        if len(x) > self.max_seq_len:
            start = np.random.randint(0, len(x) - self.max_seq_len)
            x = x[start:start + self.max_seq_len]
            y = y[start:start + self.max_seq_len]
        elif len(x) < self.max_seq_len:
            pad_len = self.max_seq_len - len(x)
            x_pad_width = ((0, pad_len), (0, 0), (0, 0), (0, 0))
            x = np.pad(x, x_pad_width, 'constant', constant_values=0)
            
            y_pad_width = ((0, pad_len), (0, 0))
            y = np.pad(y, y_pad_width, 'constant', constant_values=0)
            
        return torch.FloatTensor(x), torch.FloatTensor(difficulty_vector), torch.FloatTensor(y)

# ==============================================================================
# 4. 학습 및 검증 함수 정의
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, difficulty, y in tqdm(dataloader, desc="Training"):
        x, difficulty, y = x.to(device), difficulty.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x, difficulty)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    # 예측값과 정답값을 저장할 리스트
    # (주의: 전체를 다 저장하면 메모리가 부족할 수 있으므로, 실제로는 배치별로 계산하고 평균내는 것이 좋음)
    # 여기서는 개념 설명을 위해 전체를 저장하는 방식으로 구현
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for x, difficulty, y in tqdm(dataloader, desc="Validation"):
            x, difficulty, y = x.to(device), difficulty.to(device), y.to(device)
            outputs = model(x, difficulty)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    # 리스트를 하나의 큰 numpy 배열로 결합
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- ✨ 피크 탐지를 이용한 이진화 (핵심 수정 부분) ✨ ---
    true_events = np.zeros_like(all_labels)
    pred_events = np.zeros_like(all_outputs)
    
    # 각 배치, 각 키, 각 채널에 대해 피크 탐지 수행
    for i in range(all_labels.shape[0]): # Batch
        for j in range(all_labels.shape[2]): # Keys (4)
            for k in range(all_labels.shape[3]): # Channels (start/hold)
                # 정답 레이블에서 피크 찾기
                label_peaks, _ = find_peaks(all_labels[i, :, j, k], height=0.5, distance=3)
                true_events[i, label_peaks, j, k] = 1
                
                # 예측값에서 피크 찾기
                pred_peaks, _ = find_peaks(all_outputs[i, :, j, k], height=0.5, distance=3)
                pred_events[i, pred_peaks, j, k] = 1

    # flatten하여 전체 성능 계산
    true_flat = true_events.flatten()
    pred_flat = pred_events.flatten()
    
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    
    return total_loss / len(dataloader), f1, precision, recall

# ==============================================================================
# 5. 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # --- 데이터 준비 ---
    FILE_LIST_PATH = 'valid_files.txt'
    if not os.path.exists(FILE_LIST_PATH):
        raise FileNotFoundError(f"'{FILE_LIST_PATH}'가 없습니다. 먼저 verify_and_create_filelist.py를 실행하세요.")
    
    with open(FILE_LIST_PATH, 'r',encoding="utf-8") as f:
        all_files = [line.strip() for line in f.readlines()]
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=CONFIG["seed"])

    train_dataset = OsuDataset(train_files, CONFIG["max_seq_len"])
    val_dataset = OsuDataset(val_files, CONFIG["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"데이터 준비 완료: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개")

    # --- 모델, Loss, Optimizer 초기화 ---
    model = RhythmFormer(d_model=CONFIG["d_model"], nhead=CONFIG["nhead"], 
                         num_encoder_layers=CONFIG["num_encoder_layers"],
                         dim_feedforward=CONFIG["dim_feedforward"],
                         dropout=CONFIG["dropout"]).to(CONFIG["device"])
    
    # 가중치를 적용한 Loss (BCEWithLogitsLoss는 Sigmoid를 포함하므로 모델 마지막에 Sigmoid 제거)
    pos_weight = torch.tensor([1.5, 0.75], device=CONFIG["device"]).view(1, 1, 1, 2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    start_epoch = 0
    best_f1 = 0
    if os.path.exists(CONFIG["checkpoint_path"]):
        print(f"'{CONFIG['checkpoint_path']}'에서 체크포인트를 불러옵니다...")
        checkpoint = torch.load(CONFIG["checkpoint_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # 다음 에포크부터 시작
        best_f1 = checkpoint['best_f1']
        print(f"체크포인트 로드 완료. Epoch {start_epoch}부터 학습을 재개합니다.")
    else:
        print("체크포인트를 찾을 수 없습니다. 처음부터 학습을 시작합니다.")

    # --- 학습 시작 ---
    print(f"'{CONFIG['device']}' 장치에서 학습을 시작합니다.")

    # range 시작점을 start_epoch로 변경
    for epoch in range(start_epoch, CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss, f1, precision, recall = evaluate(model, val_loader, criterion, CONFIG["device"])

        print(f"Epoch {epoch + 1} 결과:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")
        print(f"  F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # --- ✨ 체크포인트 저장 로직 수정 ✨ ---
        # 1. 최고 성능 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            best_save_path = os.path.join(CONFIG["save_path"], f"best_model_epoch_{epoch+1}_f1_{f1:.4f}.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"✨ 최고 성능 갱신! 모델을 '{best_save_path}'에 저장했습니다.")
        
        # 2. 매 에포크가 끝날 때마다 마지막 상태 저장
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(latest_checkpoint, CONFIG["checkpoint_path"])
        print(f"Epoch {epoch + 1}의 상태를 '{CONFIG['checkpoint_path']}'에 저장했습니다.")