from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용
    allow_headers=["*"],  # 모든 헤더를 허용
)

# 더미 데이터 (선형 회귀를 위한 간단한 예제)
X_train = np.array([[1], [2], [3], [4], [5]]).reshape(-1, 1)
y_train = np.array([10, 20, 30, 40, 50])

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 입력 데이터 모델 정의
class PredictionInput(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # 입력 데이터 가공 (feature1을 예측에 사용)
    input_features = np.array([[input_data.feature1]])
    
    # 예측 수행
    prediction = model.predict(input_features)[0]
    
    return {"prediction": prediction}
