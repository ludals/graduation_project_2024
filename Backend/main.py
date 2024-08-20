from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Transformer 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, sequence_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.fc_close = nn.Linear(d_model, 1)
        self.fc_volume = nn.Linear(d_model, 1)
        self.embedding = nn.Linear(2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout(src.permute(1, 0, 2))
        tgt = self.dropout(tgt.permute(1, 0, 2))
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.permute(1, 0, 2)
        close_output = self.fc_close(transformer_out)
        volume_output = self.fc_volume(transformer_out)
        return close_output.squeeze(-1), volume_output.squeeze(-1)

# 입력 데이터 모델 정의
class PredictionInput(BaseModel):
    historical_close: list[float]
    historical_volume: list[float]

# CPU 전용 디바이스 설정
device = torch.device("cpu")

# 데이터 스케일러 정의
scaler_close = StandardScaler()
scaler_volume = StandardScaler()

# Transformer 모델 초기화
sequence_length = 30
model = TimeSeriesTransformer(d_model=128, nhead=8, num_layers=6, sequence_length=sequence_length, dropout=0.2)
model.to(device)
model.eval()

# 모델 학습 함수
def train_model(model, input_tensor, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tgt_sequence = input_tensor[-1:].unsqueeze(0)
        predicted_close, predicted_volume = model(input_tensor.unsqueeze(0), tgt_sequence)
        loss = criterion(predicted_close[:, -1], input_tensor[-1, 0]) + criterion(predicted_volume[:, -1], input_tensor[-1, 1])
        loss.backward()
        optimizer.step()

    model.eval()

# 단방향 예측 함수
def predict_next_day(model, input_sequence):
    tgt_sequence = input_sequence[-1:].unsqueeze(0)
    with torch.no_grad():
        close_output, volume_output = model(input_sequence.unsqueeze(0), tgt_sequence)
        return close_output[:, -1].item(), volume_output[:, -1].item()

# 양방향 수정 함수
def bidirectional_correction(model, input_sequence, predicted_close, predicted_volume):
    extended_sequence = torch.cat((
        input_sequence,
        torch.tensor([[predicted_close, predicted_volume]], device=device)
    ), dim=0)
    tgt_sequence = extended_sequence[-1:].unsqueeze(0)
    
    with torch.no_grad():
        corrected_close, corrected_volume = model(extended_sequence.unsqueeze(0), tgt_sequence)
        if corrected_close.size(1) > 1:
            return corrected_close[:, -2].item(), corrected_volume[:, -2].item()
        else:
            return corrected_close[:, -1].item(), corrected_volume[:, -1].item()

@app.post("/predict")
def predict(input_data: PredictionInput):
    # 입력 데이터 전처리 및 스케일링
    raw_data = list(zip(input_data.historical_close, input_data.historical_volume))
    scaled_close = scaler_close.fit_transform(np.array(input_data.historical_close).reshape(-1, 1))
    scaled_volume = scaler_volume.fit_transform(np.array(input_data.historical_volume).reshape(-1, 1))
    
    input_tensor = torch.tensor(np.hstack((scaled_close, scaled_volume)), dtype=torch.float32, device=device)

    # 모델 학습
    train_model(model, input_tensor, epochs=20)

    # 단방향 예측
    predicted_close, predicted_volume = predict_next_day(model, input_tensor)

    # 양방향 수정
    corrected_close, corrected_volume = bidirectional_correction(model, input_tensor, predicted_close, predicted_volume)

    # 예측값을 원래 스케일로 되돌리기
    predicted_close = scaler_close.inverse_transform([[predicted_close]])[0][0]
    corrected_close = scaler_close.inverse_transform([[corrected_close]])[0][0]

    # 오차 계산
    actual_close = input_tensor[-1, 0].item()
    error_close = corrected_close - actual_close

    return {
        "predicted_close": predicted_close,
        "corrected_close": corrected_close,
        "error_close": error_close,
        "predicted_volume": scaler_volume.inverse_transform([[predicted_volume]])[0][0],
        "corrected_volume": scaler_volume.inverse_transform([[corrected_volume]])[0][0],
        "error_volume": corrected_volume - input_tensor[-1, 1].item()
    }
