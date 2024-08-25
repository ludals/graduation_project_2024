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

# 예측 함수 정의
def predict_next_day(model, input_tensor):
    tgt_sequence = input_tensor[-1:].unsqueeze(0)
    with torch.no_grad():
        predicted_close = model(input_tensor.unsqueeze(0), tgt_sequence)
        return predicted_close[:, -1].item()

# 종목 리스트 로드 함수
def load_companies(filename='companies.txt'):
    with open(filename, 'r', encoding='utf-8') as file:
        companies = [line.strip() for line in file.readlines()]
    return companies

# 종목 리스트 로드
# companies = load_companies()
companies = ["028260_삼성물산_daily_data",
"207940_삼성바이오로직스_daily_data"
,"032830_삼성생명_daily_data"
,"018260_삼성에스디에스_daily_data"
,"009150_삼성전기_daily_data"]
# ,"005930_삼성전자_daily_data"]

# 모델 불러오기 및 예측
for company in companies:
    stock_file = f'processed_datas/processed_{company}.csv'
    market_file = 'processed_datas/merged_filtered_data.csv'  # 마켓 데이터 파일 경로
    model_path = f'saved_models/{company}_transformer_model.pth'
    scaler_path = f'saved_models/{company}_scaler.pkl'

    if not os.path.exists(stock_file) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Necessary files for {company} not found, skipping...")
        continue

    # CSV 파일 읽기 및 2024-08-22 데이터 필터링
    stock_data = pd.read_csv(stock_file, encoding='utf-8')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values(by='Date')  # 날짜 순으로 정렬
    
    # 2024-08-22 이전의 데이터만 사용
    stock_data = stock_data[stock_data['Date'] >= '2003-12-01']  # 2003-12-01 이후의 데이터만 사용
    data_up_to_0822 = stock_data[stock_data['Date'] <= '2024-08-22']

    if data_up_to_0822.empty:
        print(f"No sufficient data found for {company} to predict 2024-08-23.")
        continue

    # # 필요한 칼럼만 사용
    # stock_features = ['open', 'high', 'low', 'close', 'volume', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'ATR', 'OBV']

    # 마켓 데이터 로드 및 결합
    market_data = pd.read_csv(market_file, encoding='utf-8')
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.sort_values(by='Date')  # 날짜 순으로 정렬
    market_data = market_data[market_data['Date'] >= stock_data['Date'].min()]  # 종목 데이터의 시작 날짜 이후의 마켓 데이터 사용
    market_data_up_to_0822 = market_data[market_data['Date'] <= '2024-08-22']
    market_data = market_data.reset_index(drop=True)
    stock_data.set_index('Date', inplace=True)
    market_data.set_index('Date', inplace=True)

    # 병합할 때 Date를 기준으로 병합합니다
    data_up_to_0822 = pd.merge(stock_data, market_data, left_index=True, right_index=True, how='inner')

    # market_features = [
    #     'KOSPI.Open', 'KOSPI.High', 'KOSPI.Low', 'KOSPI.Close', 'KOSPI.Volume',
    #     'USD/KRW', 'CNY/KRW', 'JPY/KRW', 'EUR/KRW',
    #     'Gold (KRW)', 'Crude Oil (KRW)', 'Natural Gas (KRW)',
    #     'Dow.Close', 'NASDAQ.Close', 'SPX.Close',
    #     'VIX.Close', 'HSI.Close', 'NIKKEI.Close'
    # ]

    # 불러온 데이터 스케일링
    scaler = joblib.load(scaler_path)
    data_up_to_0822_scaled = scaler.transform(data_up_to_0822.fillna(0))
    input_tensor = torch.tensor(data_up_to_0822_scaled, dtype=torch.float32)

    # 모델 불러오기
    model = TimeSeriesTransformer(input_dim=84, d_model=128, nhead=8, num_layers=6, sequence_length=input_tensor.shape[0], dropout=0.2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 예측
    predicted_close = predict_next_day(model, input_tensor)
    predicted_close = scaler.inverse_transform([[predicted_close] + [0]*(input_tensor.shape[1]-1)])[0][0]
    
    print(f"{company} Predicted Close on 2024-08-23: {predicted_close:.2f} 원")
