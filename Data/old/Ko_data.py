import yfinance as yf
import pandas as pd

# KOSPI와 KOSDAQ 데이터 로드
kospi = yf.download('^KS11', start='2000-01-01')
kosdaq = yf.download('^KQ11', start='2000-01-01')

# 데이터프레임 통합
combined_df = pd.concat({'KOSPI': kospi, 'KOSDAQ': kosdaq}, axis=1)

# 날짜별로 정렬
combined_df.sort_index(inplace=True)

# NaN 값 처리
combined_df.ffill(inplace=True)  # 앞의 값으로 NaN 채우기

# CSV 파일로 저장
combined_df.to_csv('kospi_kosdaq_data.csv', encoding='utf-8-sig')

print("Data saved to 'kospi_kosdaq_data.csv'.")
