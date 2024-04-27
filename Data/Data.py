import pandas as pd
import FinanceDataReader as fdr

# 데이터 로드 함수
def load_data(symbol, start, end=None):
    try:
        return fdr.DataReader(symbol, start, end)
    except Exception as e:
        print(f"Error loading data for: {symbol}, {start} to {end} - {e}")
        return pd.DataFrame()

# 데이터 로드
data = {
    'KOSPI': load_data('KS11', '2020-01-01'),  # KOSPI
    'KOSDAQ': load_data('KQ11', '2020-01-01'),  # KOSDAQ
    'KOSPI 200': load_data('KS200', '2020-01-01'),  # KOSPI 200
    '다우존스': load_data('DJI', '2020-01-01'),  # 다우존스
    '나스닥': load_data('IXIC', '2020-01-01'),  # 나스닥
    'S&P500': load_data('S&P500', '2020-01-01'),  # S&P500
    'VIX지수': load_data('VIX', '2020-01-01'),  # VIX지수
    '항셍지수': load_data('HSI', '2020-01-01'),  # 항셍지수
    '닛케이': load_data('N225', '2020-01-01'),  # 닛케이
    'FTSE100': load_data('FTSE', '2020-01-01'),  # FTSE100
    'CAC 40': load_data('FCHI', '2020-01-01'),  # CAC 40
    '달러인덱스':load_data('^NYICDX', '2020-01-01'), # 달러인덱스
}

# 데이터프레임 통합
combined_df = pd.concat(data, axis=1)

# 날짜별로 정렬
combined_df.sort_index(inplace=True)

# NaN 값 처리
combined_df.fillna(method='ffill', inplace=True)  # 앞의 값으로 NaN 채우기

# CSV 파일로 저장
combined_df.to_csv('market_data.csv', encoding='utf-8-sig')

print("Data saved to 'market_data.csv'.")
