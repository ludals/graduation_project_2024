from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
import json
import joblib
import pandas as pd
import numpy as np

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 경로 설정
MODEL_PATH = "./models"  # 모델이 저장된 폴더
DATA_PATH = "./data"  # 실제 데이터 파일 경로
SCALER_X_PATH = "./data/scaler_X.pkl"  # X 스케일러 경로
SCALER_Y_PATH = "./data/scaler_y.pkl"  # y 스케일러 경로
MARKET_INDICATORS_PATH = "./data/market_indicators.csv"

# 스케일러 로드
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# DLinear 모델 정의
class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(torch.nn.Module):
    def __init__(self, input_size, seq_length, pred_length=1, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_length
        self.pred_len = pred_length
        self.individual = individual
        self.channels = input_size

        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(torch.nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(torch.nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = torch.nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = torch.nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

# 모델 로드 함수
'''def load_model(ticker):
    model_path = os.path.join(MODEL_PATH, f"{ticker}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")

    # 모델 초기화 및 가중치 로드 (seq_length=60으로 설정)
    model = DLinear(input_size=42, seq_length=60, pred_length=1, individual=False)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # 가중치 로드
    model.eval()  # 평가 모드 설정
    return model'''
# 모델 로드 함수 수정
def load_model(ticker):
    model_path = os.path.join(MODEL_PATH, f"{ticker}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")

    # 모델 초기화 및 가중치 로드 (seq_length=60으로 설정)
    model = DLinear(input_size=42, seq_length=60, pred_length=1, individual=False)
    
    # 가중치만 로드하도록 수정
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)  # 가중치 로드
    model.eval()  # 평가 모드 설정
    return model


# 예측 수행 함수 (역변환 포함)
def predict_price(model, input_data):
    print(f"Input data shape before scaling: {input_data.shape}")  # 디버깅용

    # 3차원 데이터를 2차원으로 변환 (batch_size * seq_len, features)
    input_data_reshaped = input_data.reshape(-1, input_data.shape[-1])

    # 스케일링 수행
    input_data_scaled = scaler_X.transform(input_data_reshaped)

    # 다시 3차원으로 변환 (batch_size, seq_len, features)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)  # 예측 수행
        print(f"Model output shape: {output.shape}")  # 디버깅용

        # 다차원 텐서 처리: 첫 번째 값만 사용하거나 평균 계산
        predicted_scaled = output.mean().item()

    # 예측된 값을 역변환
    predicted_price = scaler_y.inverse_transform([[predicted_scaled]])[0][0]
    
    print(f"Predicted price: {predicted_price}")
    return predicted_price

def load_and_merge_data(ticker):
    """
    각 주식 종목 데이터와 market_indicators.csv 데이터를 날짜(Date) 기준으로 병합.
    """
    # 1. 주식 종목 데이터 파일 로드
    stock_file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(stock_file_path):
        raise FileNotFoundError(f"{ticker}.csv 파일이 존재하지 않습니다.")

    stock_data = pd.read_csv(stock_file_path)

    # 2. 시장 지표 데이터 로드
    market_data = pd.read_csv(MARKET_INDICATORS_PATH)

    # 3. 날짜를 datetime 형식으로 변환 및 정렬
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    market_data['Date'] = pd.to_datetime(market_data['Date'])

    stock_data = stock_data.sort_values(by='Date', ascending=False)
    market_data = market_data.sort_values(by='Date', ascending=False)

    # 4. 날짜(Date) 기준으로 병합 (inner join)
    merged_data = pd.merge(stock_data, market_data, on='Date', how='inner')

    # 5. 피처와 타겟으로 분리
    features = merged_data.drop(columns=['Date', 'close']).values  # 피처
    target = merged_data['close'].values  # 타겟 (종가)

    return features, target

# 1. 다음 날 예측 API
# JSON 응답에서 UTF-8 인코딩 설정
@app.route('/api/predict-next-day', methods=['GET'])
def predict_next_day():
    predictions = []

    tickers = [f.split(".")[0] for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]

    for ticker in tickers:
        try:
            print(f"Processing ticker: {ticker}")

            # 모델 로드
            model = load_model(ticker)

            # 데이터 병합 및 준비
            features, _ = load_and_merge_data(ticker)

            # feature 개수 확인 및 조정
            if features.shape[1] < 42:
                missing_features = 42 - features.shape[1]
                features = np.hstack([features, np.zeros((features.shape[0], missing_features))])

            # 입력 데이터 생성 (마지막 60개 행 사용)
            input_data = features[-60:].reshape(1, 60, -1)

            # 예측 수행
            predicted_price = predict_price(model, input_data)
            predictions.append({"ticker": ticker, "predictedNextDayPrice": predicted_price})

        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    # JSON 응답 생성 시 ensure_ascii=False로 설정
    return jsonify(predictions), 200, {'Content-Type': 'application/json; charset=utf-8'}

'''@app.route('/api/predict-next-day', methods=['GET'])
def predict_next_day():
    predictions = []

    # data 폴더 내의 모든 주식 종목 CSV 파일의 이름을 추출
    tickers = [f.split(".")[0] for f in os.listdir(DATA_PATH) if f.endswith(".csv") and f != "market_indicators.csv"]

    for ticker in tickers:
        try:
            model = load_model(ticker)  # 각 종목에 맞는 모델 로드

            # 5. 데이터 병합 및 준비 (각 종목별 데이터 병합)
            features, _ = load_and_merge_data(ticker)

            # 6. 입력 데이터 생성 (마지막 60개 행을 사용)
            input_data = features[-60:].reshape(1, 60, -1)  # (1, 60, feature_size)

            # 7. 예측 수행
            predicted_price = predict_price(model, input_data)
            predictions.append({"ticker": ticker, "predictedNextDayPrice": predicted_price})

        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200
'''
"""@app.route('/api/predict-next-day', methods=['GET'])
def predict_next_day():
    predictions = []

    tickers = [f.split(".")[0] for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]

    for ticker in tickers:
        try:
            model = load_model(ticker)  # 모델 로드

            # 입력 데이터 생성 (batch_size=1, seq_length=60, features=10)
            input_data = torch.randn(1, 60, 42).numpy()

            predicted_price = predict_price(model, input_data)
            predictions.append({"ticker": ticker, "predictedNextDayPrice": predicted_price})

        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200
    """


# 2. 종목 상세 데이터 API (실제 데이터 + 예측 데이터)
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        model = load_model(ticker)  # 모델 로드
        predicted_prices = [predict_price(model) for _ in range(10)]  # 10일 예측 수행

        # 실제 데이터 로드
        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Stock data file not found"}), 500

        with open(DATA_PATH) as f:
            actual_data = json.load(f)
        actual_prices = actual_data.get(ticker, [])

        return jsonify({
            "ticker": ticker,
            "actualPrices": actual_prices,
            "predictedPrices": predicted_prices
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
