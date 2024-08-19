from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn

app = FastAPI()

# Transformer 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, sequence_length):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc_close = nn.Linear(d_model, 1)  # 종가 예측 레이어
        self.fc_volume = nn.Linear(d_model, 1)  # 거래량 예측 레이어
        self.embedding = nn.Linear(2, d_model)  # 2D 입력 (종가와 거래량) -> d_model 차원 임베딩

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, d_model) -> (sequence_length, batch_size, d_model)
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.permute(1, 0, 2)  # (sequence_length, batch_size, d_model) -> (batch_size, sequence_length, d_model)
        close_output = self.fc_close(transformer_out)  # (batch_size, sequence_length, 1)
        volume_output = self.fc_volume(transformer_out)  # (batch_size, sequence_length, 1)
        return close_output.squeeze(-1), volume_output.squeeze(-1)

# 입력 데이터 모델 정의
class PredictionInput(BaseModel):
    historical_close: list[float]
    historical_volume: list[float]

# Transformer 모델 초기화 (임의의 학습된 모델 사용)
sequence_length = 30
model = TimeSeriesTransformer(d_model=64, nhead=8, num_layers=4, sequence_length=sequence_length)
model.eval()  # 예측 모드로 설정

# 단방향 예측 함수
def predict_next_day(model, input_sequence):
    """
    단방향 예측: 다음날 종가와 거래량 예측
    """
    with torch.no_grad():
        close_output, volume_output = model(input_sequence.unsqueeze(0))
        return close_output[:, -1].item(), volume_output[:, -1].item()

# 양방향 수정 함수
def bidirectional_correction(model, input_sequence, predicted_close, predicted_volume):
    """
    양방향 수정: 예측된 미래값을 바탕으로 현재값 재평가
    """
    extended_sequence = torch.cat((
        input_sequence,
        torch.tensor([[predicted_close, predicted_volume]])
    ), dim=0)
    
    with torch.no_grad():
        corrected_close, corrected_volume = model(extended_sequence.unsqueeze(0))
        return corrected_close[:, -2].item(), corrected_volume[:, -2].item()

@app.post("/predict")
def predict(input_data: PredictionInput):
    # 입력 데이터 전처리
    input_tensor = torch.tensor(
        list(zip(input_data.historical_close, input_data.historical_volume)),
        dtype=torch.float32
    )

    # Step 1: 단방향 예측
    predicted_close, predicted_volume = predict_next_day(model, input_tensor)

    # Step 2: 양방향 수정
    corrected_close, corrected_volume = bidirectional_correction(model, input_tensor, predicted_close, predicted_volume)

    # Step 3: 오차 계산 (실제 데이터와 비교할 수 있는 상황을 가정)
    actual_close = input_tensor[-1, 0].item()
    actual_volume = input_tensor[-1, 1].item()

    error_close = corrected_close - actual_close
    error_volume = corrected_volume - actual_volume

    # 여기서 오차를 이용해 모델을 업데이트할 수 있습니다 (현재는 더미 코드로 표시)
    # 예: optimizer.zero_grad(), loss.backward(), optimizer.step() 등

    return {
        "predicted_close": predicted_close,
        "corrected_close": corrected_close,
        "error_close": error_close,
        "predicted_volume": predicted_volume,
        "corrected_volume": corrected_volume,
        "error_volume": error_volume
    }
