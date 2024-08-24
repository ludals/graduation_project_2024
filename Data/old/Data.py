import pandas as pd
import FinanceDataReader as fdr

# 데이터 로드 함수
def load_data(symbol, start='2020-01-01', end=None):
    try:
        df = fdr.DataReader(symbol, start, end)
        print(f"{symbol}: {df.index.min()} to {df.index.max()}, Total records: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data for: {symbol}, {start} to {end} - {e}")
        return pd.DataFrame()

# 데이터 로드
symbols = {
    'KOSPI': 'KS11',  # KOSPI
    'KOSDAQ': 'KQ11',  # KOSDAQ
    'KOSPI 200': 'KS200',  # KOSPI 200
    '다우존스': 'DJI',  # 다우존스
    '나스닥': 'IXIC',  # 나스닥
    'S&P500': 'S&P500',  # S&P500
    'VIX지수': 'VIX',  # VIX지수
    '항셍지수': 'HSI',  # 항셍지수
    '닛케이': 'N225',  # 닛케이
    'FTSE100': 'FTSE',  # FTSE100
    'CAC 40': 'FCHI',  # CAC 40
    '달러인덱스': '^NYICDX',  # 달러인덱스
}

# 데이터프레임 통합
data = {name: load_data(symbol, '2020-01-01') for name, symbol in symbols.items()}
combined_df = pd.concat(data, axis=1)

# 날짜별로 정렬
combined_df.sort_index(inplace=True)

# NaN 값 처리
combined_df.ffill(inplace=True)  # 앞의 값으로 NaN 채우기
  # 앞의 값으로 NaN 채우기

# CSV 파일로 저장
combined_df.to_csv('market_data_2.csv', encoding='utf-8-sig')

print("Data saved to 'market_data.csv'.")