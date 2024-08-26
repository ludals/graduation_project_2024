import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os


# Transformer 모델 정의
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, sequence_length, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout,
                                          batch_first=True)
        self.fc = nn.Linear(d_model, input_dim)  # 모든 피처를 예측하도록 설정
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout(src)
        tgt = self.dropout(tgt)
        transformer_out = self.transformer(src, tgt)
        output = self.fc(transformer_out)
        return output.squeeze(-1)


# 예측 함수 정의 (종가만 반환)
def predict_next_day(model, input_tensor):
    tgt_sequence = input_tensor[-1:].unsqueeze(0)
    with torch.no_grad():
        predicted = model(input_tensor.unsqueeze(0), tgt_sequence)
        return predicted[:, -1, 3]  # 종가(첫 번째 피처)만 반환


# 양방향 예측 함수 정의 (종가만 반환)
def bidirectional_correction(model, input_tensor, predicted_close):
    # 종가 예측 값만 입력 시퀀스에 추가
    predicted_full = torch.cat((predicted_close.unsqueeze(0), input_tensor[-1, 1:].unsqueeze(0)), dim=-1)
    predicted_full = predicted_full.unsqueeze(0)  # 차원 맞추기
    extended_sequence = torch.cat((input_tensor.unsqueeze(0), predicted_full), dim=1)  # 차원 맞추기
    tgt_sequence = extended_sequence[:, -1:, :]  # 마지막 타겟 시퀀스 추출

    with torch.no_grad():
        corrected = model(extended_sequence, tgt_sequence)
        return corrected[:, -1, 3]  # 수정된 종가만 반환


# 모델 학습 함수
def train_model(model, input_tensor, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tgt_sequence = input_tensor[-1:].unsqueeze(0)
        predicted = model(input_tensor.unsqueeze(0), tgt_sequence)
        loss = criterion(predicted[:, -1, :], input_tensor[-1, :].unsqueeze(0))  # 크기 맞추기
        loss.backward()
        optimizer.step()

        # 현재 epoch와 loss 출력
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()


# CSV 파일 읽기 및 데이터 전처리
def load_and_preprocess_data(stock_file, market_file):
    # CSV 파일 읽기
    stock_data = pd.read_csv(stock_file, encoding='utf-8')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values(by='Date')
    stock_data = stock_data[stock_data['Date'] >= '2003-12-01']

    market_data = pd.read_csv(market_file, encoding='utf-8')
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.sort_values(by='Date')
    market_data = market_data[market_data['Date'] >= stock_data['Date'].min()]
    market_data = market_data.reset_index(drop=True)
    stock_data.set_index('Date', inplace=True)
    market_data.set_index('Date', inplace=True)

    # 데이터 병합
    merged_data = pd.merge(stock_data, market_data, left_index=True, right_index=True, how='inner')

    # NaN 값 제거 또는 대체
    merged_data.fillna(0, inplace=True)  # NaN 값을 0으로 대체하거나 다른 값으로 대체

    # 데이터 표준화
    scaler = StandardScaler()
    merged_data_np = scaler.fit_transform(merged_data.values)

    # numpy 배열을 torch tensor로 변환
    return torch.tensor(merged_data_np, dtype=torch.float32), scaler

# 모델 저장 함수
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

# 종목 리스트
def load_companies(filename='companies.txt'):
    with open(filename, 'r', encoding='utf-8') as file:
        companies = [line.strip() for line in file.readlines()]
    return companies

# 스케일 복원
def inverse_transform_close(scaler, scaled_value):
    mean = scaler.mean_[3]  # 'Close' 피처의 평균
    scale = scaler.scale_[3]  # 'Close' 피처의 표준편차
    return scaled_value * scale + mean


# 종목 리스트 로드
companies = [
"ti_005930_삼성전자_daily_data"]

market_file = 'processed_datas/merged_filtered_data.csv'

# 모델 학습 및 저장
for company in companies:
    stock_file = f'{company}.csv'
    if not os.path.exists(stock_file):
        print(f"File {stock_file} not found, skipping...")
        continue

    print(f"Processing {company}...")

    # 입력 텐서와 스케일러 가져오기
    input_tensor, scaler = load_and_preprocess_data(stock_file, market_file)

    # 모델 초기화
    input_dim = input_tensor.shape[1]  # 피처 개수
    sequence_length = input_tensor.shape[0]  # 데이터 개수
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=128, nhead=8, num_layers=6,
                                  sequence_length=sequence_length, dropout=0.2)
    model.to(torch.device("cpu"))

    # 모델 학습
    train_model(model, input_tensor, epochs=40)

    # 단방향 예측
    predicted_close = predict_next_day(model, input_tensor)

    # 예측된 close 값을 스케일링 복원
    predicted_close_np = predicted_close.numpy().reshape(-1, 1)


    restored_close = inverse_transform_close(scaler, predicted_close_np)

    # 양방향 수정
    corrected_close = bidirectional_correction(model, input_tensor, predicted_close)

    # 수정된 close 값을 스케일링 복원
    corrected_close_np = corrected_close.numpy().reshape(-1, 1)
    restored_corrected_close = inverse_transform_close(scaler, corrected_close_np)[0, 0]

    # 모델 저장
    model_path = f'./saved_models/{company}_transformer_model.pth'
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    print(f"{company} Predicted Close on 2024-08-23: {restored_corrected_close:.2f} 원")

print("All models processed and saved.")
