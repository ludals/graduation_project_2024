from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # CORS를 위한 가져오기
from pydantic import BaseModel  # BaseModel 가져오기
import torch
import torch.nn as nn

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요한 경우 허용할 도메인을 명시
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Transformer 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, sequence_length):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc_close = nn.Linear(d_model, 1)  # 종가 예측 레이어
        self.fc_volume = nn.Linear(d_model, 1)  # 거래량 예측 레이어
        self.embedding = nn.Linear(2, d_model)  # 2D 입력 (종가와 거래량) -> d_model 차원 임베딩

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)  # (batch_size, sequence_length, d_model) -> (sequence_length, batch_size, d_model)
        tgt = tgt.permute(1, 0, 2)
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.permute(1, 0, 2)  # (sequence_length, batch_size, d_model) -> (batch_size, sequence_length, d_model)
        close_output = self.fc_close(transformer_out)  # (batch_size, sequence_length, 1)
        volume_output = self.fc_volume(transformer_out)  # (batch_size, sequence_length, 1)
        return close_output.squeeze(-1), volume_output.squeeze(-1)

# CPU 전용 디바이스 설정
device = torch.device("cpu")

# Transformer 모델 초기화 (임의의 학습된 모델 사용)
sequence_length = 30
model = TimeSeriesTransformer(d_model=64, nhead=8, num_layers=4, sequence_length=sequence_length)
model.to(device)  # 모델을 CPU로 이동
model.eval()  # 예측 모드로 설정

# 단방향 예측 함수
def predict_next_day(model, input_sequence):
    """
    단방향 예측: 다음날 종가와 거래량 예측
    """
    tgt_sequence = input_sequence[-1:].unsqueeze(0)  # 마지막 값을 tgt로 설정
    with torch.no_grad():
        close_output, volume_output = model(input_sequence.unsqueeze(0), tgt_sequence)
        # 스케일링 추가 (0 ~ 100,000 원 범위로 가정)
        scaled_close = torch.sigmoid(close_output) * 100000
        scaled_volume = torch.sigmoid(volume_output) * 100000
        return scaled_close[:, -1].item(), scaled_volume[:, -1].item()

# 양방향 수정 함수
def bidirectional_correction(model, input_sequence, predicted_close, predicted_volume):
    """
    양방향 수정: 예측된 미래값을 바탕으로 현재값 재평가
    """
    extended_sequence = torch.cat((
        input_sequence,
        torch.tensor([[predicted_close, predicted_volume]], device=device)
    ), dim=0)
    tgt_sequence = extended_sequence[-1:].unsqueeze(0)  # 새로 추가된 값을 tgt로 설정
    
    with torch.no_grad():
        corrected_close, corrected_volume = model(extended_sequence.unsqueeze(0), tgt_sequence)
        # 스케일링 추가 (0 ~ 100,000 원 범위로 가정)
        scaled_close = torch.sigmoid(corrected_close) * 100000
        scaled_volume = torch.sigmoid(corrected_volume) * 100000
        if scaled_close.size(1) > 1:  # 시퀀스 길이가 충분히 긴지 확인
            return scaled_close[:, -2].item(), scaled_volume[:, -2].item()
        else:
            return scaled_close[:, -1].item(), scaled_volume[:, -1].item()