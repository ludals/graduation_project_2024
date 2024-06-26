import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # DateTime 컬럼을 datetime 객체로 변환
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # 데이터 정렬
    df = df.sort_values('DateTime')

    # 필요한 열 선택
    df = df[['DateTime', 'Close', 'Open', 'Low', 'High', 'Volume']]
    # 인덱스를 DateTime으로 설정
    df.set_index('DateTime', inplace=True)

    df[['Close', 'Open', 'Low', 'High', 'Volume']] = df[['Close', 'Open', 'Low', 'High', 'Volume']].abs()


    return df

# 시계열 데이터 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length][0]  # 종가를 예측 대상으로 설정
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# LSTM 모델 정의
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 메인 실행 함수
def main():
    file_path = "005930_60min_data.csv"
    seq_length = 60  # 예: 60시간의 데이터를 사용하여 다음 시간의 종가를 예측

    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(file_path)

    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # 시계열 데이터 생성
    X, y = create_sequences(scaled_data, seq_length)

    # 학습/테스트 데이터 분할
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM 모델 정의 및 학습
    model = build_model((seq_length, X.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # 모델 평가
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # 6월 27일 09시 데이터 예측 (가정: 마지막 테스트 데이터를 사용하여 다음 시간을 예측)
    predicted_price = model.predict(X_test[-1].reshape(1, seq_length, X.shape[2]))
    predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], df.shape[1] - 1))), axis=1))[:, 0]

    print(f"Predicted price for June 27, 09:00: {predicted_price[0]}")

if __name__ == "__main__":
    main()
