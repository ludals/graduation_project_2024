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
        self.fc = nn.Linear(d_model, input_dim)  # 모든 피처를 예측하도록 설정
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout(src.permute(1, 0, 2))
        tgt = self.dropout(tgt.permute(1, 0, 2))
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.permute(1, 0, 2)
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
    extended_sequence = torch.cat((input_tensor, predicted_full.unsqueeze(0)), dim=0)
    tgt_sequence = extended_sequence[-1:].unsqueeze(0)
    
    with torch.no_grad():
        corrected = model(extended_sequence.unsqueeze(0), tgt_sequence)
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
        loss = criterion(predicted[:, -1, :], input_tensor[-1, :])
        loss.backward()
        optimizer.step()

    model.eval()

# CSV 파일 읽기 및 데이터 전처리
def load_and_preprocess_data(stock_file, market_file):
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

    merged_data = pd.merge(stock_data, market_data, left_index=True, right_index=True, how='inner')
    
    return torch.tensor(merged_data, dtype=torch.float32)

# 모델 저장 함수
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

# 종목 리스트
def load_companies(filename='companies.txt'):
    with open(filename, 'r', encoding='utf-8') as file:
        companies = [line.strip() for line in file.readlines()]
    return companies

# 종목 리스트 로드
companies = [
"ti_005930_삼성전자_daily_data"]

market_file = 'processed_datas/merged_filtered_data.csv'

# 모델 학습 및 저장
for company in companies:
    stock_file = f'processed_datas/processed_{company}.csv'
    if not os.path.exists(stock_file):
        print(f"File {stock_file} not found, skipping...")
        continue

    print(f"Processing {company}...")

    input_tensor = load_and_preprocess_data(stock_file, market_file)
    
    # 모델 초기화
    input_dim = input_tensor.shape[1]  # 피처 개수
    sequence_length = input_tensor.shape[0]  # 데이터 개수
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=128, nhead=8, num_layers=6, sequence_length=sequence_length, dropout=0.2)
    model.to(torch.device("cpu"))

    # 모델 학습
    train_model(model, input_tensor, epochs=20)

    # 단방향 예측
    predicted_close = predict_next_day(model, input_tensor)

    # 양방향 수정
    corrected_close = bidirectional_correction(model, input_tensor, predicted_close)
    

    # 모델 저장
    model_path = f'./saved_models/{company}_transformer_model.pth'
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    print(f"{company} Predicted Close on 2024-08-23: {corrected_close.item():.2f} 원")

print("All models processed and saved.")
