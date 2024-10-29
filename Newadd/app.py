from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
import json

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 경로 설정
MODEL_PATH = "./models"  # 모델이 저장된 폴더
DATA_PATH = "./data/stock_data.json"  # 실제 데이터 파일 경로
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
def load_model(ticker):
    model_path = os.path.join(MODEL_PATH, f"{ticker}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")

    # 모델 초기화 및 가중치 로드 (seq_length=60으로 설정)
    model = DLinear(input_size=10, seq_length=60, pred_length=1, individual=False)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # 가중치 로드
    model.eval()  # 평가 모드 설정
    return model

# 예측 수행 함수
# 예측 수행 함수
def predict_next_day_price(model, input_data):
    with torch.no_grad():
        output = model(input_data)
        # 첫 번째 예측 값만 선택 (예: 출력이 [1, pred_length, channel] 형태일 경우)
        return output[0, 0, 0].item()  # 필요한 경우 인덱스를 조정

# 예측 수행 함수
# def predict_price(model):
#     input_data = torch.randn(1, 10)  # 임시 입력 데이터 (10차원)
#     with torch.no_grad():
#         return model(input_data).item()

# 1. 다음 날 예측 API
@app.route('/api/predict-next-day', methods=['GET'])
def predict_next_day():
    predictions = []

    # models 폴더에서 모든 .pth 파일 이름 추출
    tickers = [f.split(".")[0] for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]

    for ticker in tickers:
        try:
            model = load_model(ticker)  # 모델 로드
            input_data = torch.randn(1, 60, 10)  # 임시 입력 데이터 (실제 입력 형식에 맞게 수정 필요)
            predicted_price = predict_next_day_price(model, input_data)
            predictions.append({"ticker": ticker, "predictedNextDayPrice": predicted_price})
        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200
# 1. 모든 종목의 예측 가격 제공 API
@app.route('/api/predicted-prices', methods=['GET'])
def get_predicted_prices():
    predictions = []

    # models 폴더에서 모든 .pth 파일 이름 추출
    tickers = [f.split(".")[0] for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]

    for ticker in tickers:
        try:
            model = load_model(ticker)  # 모델 로드
            predicted_price = predict_price(model)  # 예측 수행
            predictions.append({"ticker": ticker, "predictedPrice": predicted_price})
        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200

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

"""from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
import random

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정

# 모델 로드 함수
def load_model(ticker):
    model_path = f"./models/{ticker}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")

    model = torch.load(model_path)  # 학습된 모델 불러오기
    model.eval()  # 모델을 평가 모드로 설정
    return model

# 예측 수행 함수
def predict_price(model):
    input_data = torch.randn(1, 10)  # 임시 입력 데이터
    with torch.no_grad():
        return model(input_data).item()

# 1. 모든 종목의 예측 가격 제공 API
@app.route('/api/predicted-prices', methods=['GET'])
def get_predicted_prices():
    ###stocks = ["AAPL", "TSLA", "GOOGL"]
    stocks = []
    predictions = []

    for ticker in stocks:
        try:
            model = load_model(ticker)
            predicted_price = predict_price(model)
            predictions.append({"ticker": ticker, "predictedPrice": predicted_price})
        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200

# 2. 종목 상세 데이터 API (실제 데이터 + 예측 데이터)
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker = request.args.get('ticker')

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        model = load_model(ticker)
        predicted_prices = [predict_price(model) for _ in range(10)]  # 10일 예측

        # 임의의 실제 가격 데이터 생성
        actual_prices = [random.uniform(100, 200) for _ in range(10)]

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
"""

"""from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
import json
#import random

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 모델 로드 함수
def load_model(ticker):
    model_path = f"./models/{ticker}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")
    
    model = torch.load(model_path)
    model.eval()
    return model

# 예측 수행 함수
def predict_price(model):
    input_data = torch.randn(1, 10)  # 임시 입력 데이터
    with torch.no_grad():
        return model(input_data).item()

# 1. 모든 종목의 예측 가격 제공 API
@app.route('/api/predicted-prices', methods=['GET'])
def get_predicted_prices():
    stocks = ["AAPL", "TSLA", "GOOGL"]
    predictions = []

    for ticker in stocks:
        try:
            model = load_model(ticker)
            predicted_price = predict_price(model)
            predictions.append({"ticker": ticker, "predictedPrice": predicted_price})
        except FileNotFoundError as e:
            predictions.append({"ticker": ticker, "error": str(e)})

    return jsonify(predictions), 200

# 2. 종목 상세 데이터 API (실제 데이터 + 예측 데이터)
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        model = load_model(ticker)
        predicted_prices = [predict_price(model) for _ in range(10)]

        # 실제 데이터 로드 (임의의 JSON 파일 사용)
        with open('./data/stock_data.json') as f:
            actual_data = json.load(f)
        actual_prices = actual_data.get(ticker, [])

        return jsonify({
            "ticker": ticker,
            "actualPrices": actual_prices,
            "predictedPrices": predicted_prices
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
