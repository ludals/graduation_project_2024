import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import numpy as np
import time

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("yummy")
else:
    print("not yummy")
torch.backends.cudnn.benchmark = True

# Transformer 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.dropout(self.embedding(src))
        tgt = self.dropout(self.embedding(tgt))
        transformer_out = self.transformer(src, tgt)
        return self.fc(transformer_out).squeeze(-1)

# CSV 파일 읽기 및 데이터 전처리
def load_and_preprocess_data(stock_file: str, market_file: str) -> Tuple[torch.Tensor, StandardScaler]:
    stock_data = pd.read_csv(stock_file, encoding='utf-8')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[stock_data['Date'] >= '2015-12-01'].sort_values(by='Date')

    market_data = pd.read_csv(market_file, encoding='utf-8')
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data[market_data['Date'] >= stock_data['Date'].min()].sort_values(by='Date')

    merged_data = pd.merge(stock_data, market_data, on='Date', how='inner').fillna(0)

    # 'Date' 열을 인덱스로 설정
    merged_data.set_index('Date', inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_data.values)
    return torch.tensor(scaled_data, dtype=torch.float32).to(device), scaler

# 양방향 예측 보정 함수
def bidirectional_correction(model: nn.Module, input_tensor: torch.Tensor, predicted_close: torch.Tensor) -> torch.Tensor:
    predicted_full = torch.cat((predicted_close.unsqueeze(0), input_tensor[-1, 1:].unsqueeze(0)), dim=-1).unsqueeze(0)
    extended_sequence = torch.cat((input_tensor.unsqueeze(0), predicted_full), dim=1)
    tgt_sequence = extended_sequence[:, -1:, :]

    with torch.no_grad():
        corrected = model(extended_sequence, tgt_sequence)
    return corrected[:, -1, 3]

# Backcasting을 이용한 예측 보정 함수
def update_prediction_with_backcast(model: nn.Module, input_tensor: torch.Tensor, predicted_value: torch.Tensor, target_index: int) -> torch.Tensor:
    if target_index >= len(input_tensor):
        target_index = len(input_tensor) - 1

    predicted_full = torch.cat((predicted_value.unsqueeze(0), input_tensor[target_index, 1:].unsqueeze(0)), dim=-1).unsqueeze(0)
    extended_sequence = torch.cat((input_tensor[:target_index].unsqueeze(0), predicted_full), dim=1)
    tgt_sequence = extended_sequence[:, -1:, :]

    with torch.no_grad():
        backcasted_value = model(extended_sequence[:, :-1, :], tgt_sequence)
        backcasted_close = backcasted_value[:, -1, 3]

    correction_factor = input_tensor[target_index - 1, 3] - backcasted_close
    return predicted_value + correction_factor

# 모델 학습 함수
def train_model(model: nn.Module, input_tensor: torch.Tensor, epochs: int = 20, batch_size: int = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Epoch [{epoch + 1}/{epochs}] started at {current_time}")

        total_loss = 0.0

        for t in range(1, input_tensor.shape[0]):
            batch_input = input_tensor[max(0, t-batch_size):t].unsqueeze(0)
            tgt_sequence = input_tensor[t-1:t].unsqueeze(0)
            predicted = model(batch_input, tgt_sequence)[:, -1, 3]

            if t > 1:
                predicted = update_prediction_with_backcast(model, input_tensor[:t], predicted, t)

            loss = criterion(predicted, input_tensor[t, 3].unsqueeze(0))
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}')

    model.eval()

# 예측 함수
def predict_next_day(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    tgt_sequence = input_tensor[-1:].unsqueeze(0)
    with torch.no_grad():
        predicted = model(input_tensor.unsqueeze(0), tgt_sequence)
    return predicted[:, -1, 3]

# 모델 저장 함수
def save_model(model: nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)

# 스케일 복원 함수
def inverse_transform_close(scaler: StandardScaler, scaled_value: np.ndarray) -> np.ndarray:
    return scaled_value * scaler.scale_[3] + scaler.mean_[3]

# 메인 함수
def main(companies: List[str], market_file: str):
    print("in main func Current working directory:", os.getcwd())

    for company in companies:
        stock_file = os.path.join(os.getcwd(), f'{company}.csv')
        print(f"Full file path: {stock_file}")
        if not os.path.exists(stock_file):
            print(f"File {stock_file} not found, skipping...")
            continue

        print(f"Processing {company}...")
        input_tensor, scaler = load_and_preprocess_data(stock_file, market_file)

        model = TimeSeriesTransformer(input_dim=input_tensor.shape[1], d_model=128, nhead=8, num_layers=6, dropout=0.2).to(device)
        train_model(model, input_tensor, epochs=20)

        predicted_close = predict_next_day(model, input_tensor)
        restored_close = inverse_transform_close(scaler, predicted_close.cpu().numpy().reshape(-1, 1))
        print(f"{company} Predicted Close on 2024-08-23: {restored_close[0, 0]:.2f} 원")

        model_path = f'{company}_transformer_model.pth'
        save_model(model, model_path)
        print(f"Model saved to {model_path}")

    print("All models processed and saved.")

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    companies = ["ti_005930_삼성전자_daily_data"]
    market_file = os.path.join(os.getcwd(), 'processed_datas/merged_filtered_data.csv')
    main(companies, market_file)
