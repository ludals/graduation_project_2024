"""import torch
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 생성 및 학습 (예제 데이터 사용)
model = DLinear()
dummy_input = torch.randn(100, 10)  # 임시 학습 데이터
dummy_output = torch.randn(100, 1)  # 임시 레이블

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 간단한 학습 루프
for epoch in range(100):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_output)
    loss.backward()
    optimizer.step()
"""
# 모델 저장
#torch.save(model, './models/AAPL.pth')
#torch.save(model, './models/TSLA.pth')
#torch.save(model, './models/GOOGL.pth')

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from external_models.dlinear_model import DLinear  # 외부에서 정의된 D-Linear 모델 사용, 나중에 파일 구조 다시 설정하면 해결 가능

# 데이터 및 모델 경로 설정
DATA_PATH = "./data"      # 주식 데이터가 있는 폴더
MODEL_PATH = "./models"   # 학습된 모델이 저장될 폴더

# 모델 학습 및 저장 함수
def train_and_save_model(ticker, input_size=10, batch_size=32, epochs=50):
    # 1. 주식 데이터 로드
    file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")

    df = pd.read_csv(file_path)

    # 2. 입력과 출력 데이터로 분리
    inputs = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)  # 입력 데이터 (Features)
    targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)  # 출력 데이터 (Target)

    # 3. DataLoader 생성 (배치 학습 지원)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. D-Linear 모델 초기화
    model = DLinear(input_size=input_size, output_size=1)
    criterion = nn.MSELoss()  # 손실 함수 (Mean Squared Error)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 최적화 알고리즘 (Adam)

    # 5. 모델 학습 루프
    model.train()  # 학습 모드로 설정
    for epoch in range(epochs):
        total_loss = 0  # 한 epoch 당 누적 손실값
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()  # 이전 기울기 초기화
            outputs = model(batch_inputs)  # 예측 수행
            loss = criterion(outputs, batch_targets)  # 손실 계산
            loss.backward()  # 역전파 수행
            optimizer.step()  # 가중치 업데이트
            total_loss += loss.item()  # 손실 누적

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    # 6. 학습된 모델 저장
    model_path = os.path.join(MODEL_PATH, f"{ticker}.pth")
    torch.save(model.state_dict(), model_path)  # 가중치만 저장
    print(f"Model for {ticker} saved at {model_path}")

# 모든 CSV 파일을 탐색하여 모델 학습 및 저장
def main():
    # 1. data 폴더에서 모든 CSV 파일 이름 가져오기
    tickers = [f.split(".")[0] for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

    # 2. 각 종목에 대해 모델 학습 및 저장 수행
    for ticker in tickers:
        try:
            print(f"Training model for {ticker}...")
            train_and_save_model(ticker)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error while training {ticker}: {str(e)}")

# 메인 함수 실행
if __name__ == "__main__":
    main()
