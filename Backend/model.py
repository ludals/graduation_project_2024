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
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.fc_close = nn.Linear(d_model, 1)
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout(src.permute(1, 0, 2))
        tgt = self.dropout(tgt.permute(1, 0, 2))
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.permute(1, 0, 2)
        close_output = self.fc_close(transformer_out)
        return close_output.squeeze(-1)

# 모델 학습 함수
def train_model(model, input_tensor, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tgt_sequence = input_tensor[-1:].unsqueeze(0)
        predicted_close = model(input_tensor.unsqueeze(0), tgt_sequence)
        loss = criterion(predicted_close[:, -1], input_tensor[-1, 0])
        loss.backward()
        optimizer.step()

    model.eval()

# CSV 파일 읽기 및 데이터 전처리
def load_and_preprocess_data(stock_file, market_file):
    # 종목 데이터 로드
    stock_data = pd.read_csv(stock_file, encoding='cp949')
    stock_features = ['open', 'high', 'low', 'close', 'volume', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'ATR', 'OBV']
    stock_data = stock_data[stock_features].fillna(0)  # 결측값 처리
    
    # 마켓 데이터 로드
    market_data = pd.read_csv(market_file, encoding='cp949')
    market_features = [
        'KOSPI.Open', 'KOSPI.High', 'KOSPI.Low', 'KOSPI.Close', 'KOSPI.Volume',
        'USD/KRW', 'CNY/KRW', 'JPY/KRW', 'EUR/KRW',
        'Gold (KRW)', 'Crude Oil (KRW)', 'Natural Gas (KRW)',
        '다우존스.Close', '나스닥.Close', 'S&P500.Close',
        'VIX지수.Close', '항셍지수.Close', '닛케이.Close'
    ]
    market_data = market_data[market_features].fillna(0)  # 결측값 처리

    # 날짜 기준으로 병합
    merged_data = pd.merge(stock_data, market_data, left_index=True, right_index=True)

    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_data)
    
    return torch.tensor(scaled_data, dtype=torch.float32), scaler

# 모델 저장 함수
def save_model(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

# 종목 리스트
def load_companies(filename='companies.txt'):
    with open(filename, 'r', encoding='utf-8') as file:
        companies = [line.strip() for line in file.readlines()]
    return companies

# 종목 리스트 로드
companies = load_companies()

market_file = 'merged_filtered_data.csv'  # 마켓 데이터 파일 경로

# 모델 학습 및 저장
for company in companies:
    stock_file = f'processed_{company}.csv'  # 불러올 종목 CSV 파일 이름
    if not os.path.exists(stock_file):
        print(f"File {stock_file} not found, skipping...")
        continue

    print(f"Processing {company}...")

    input_tensor, scaler = load_and_preprocess_data(stock_file, market_file)
    
    # 모델 초기화
    input_dim = input_tensor.shape[1]  # 피처 개수에 따라 input_dim 설정
    sequence_length = input_tensor.shape[0]  # 시퀀스 길이 설정
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=4, sequence_length=sequence_length, dropout=0.2)
    model.to(torch.device("cpu"))

    # 모델 학습
    train_model(model, input_tensor, epochs=20)

    # 모델 저장
    model_path = f'{company}_transformer_model.pth'
    scaler_path = f'{company}_scaler.pkl'
    save_model(model, scaler, model_path, scaler_path)
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}")

print("All models processed and saved.")
