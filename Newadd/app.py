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

# 모델 로드 함수
def load_model(ticker):
    model_path = os.path.join(MODEL_PATH, f"{ticker}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found")

    model = torch.load(model_path, map_location=torch.device('cpu'))  # CPU 모드에서 로드
    model.eval()  # 평가 모드 설정
    return model

# 예측 수행 함수
def predict_next_day_price(model, input_data):
    with torch.no_grad():
        return model(input_data).item()
# 예측 수행 함수
# def predict_price(model):
#     input_data = torch.randn(1, 10)  # 임시 입력 데이터 (10차원)
#     with torch.no_grad():
#         return model(input_data).item()

# 1. 다음 날 예측 API
@app.route('/api/predict-next-day', methods=['GET'])
def predict_next_day():
    predictions = []  # 각 ticker의 예측값을 저장할 리스트

    # models 폴더에서 모든 .pth 파일 이름 추출
    tickers = [f.split(".")[0] for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]

    for ticker in tickers:
        try:
            model = load_model(ticker)  # 모델 로드
            input_data = torch.randn(1, 10)  # 임시 입력 데이터
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
