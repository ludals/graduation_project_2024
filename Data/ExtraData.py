import FinanceDataReader as fdr
import pandas as pd

# 600일치 데이터 추출을 위한 날짜 설정
end_date = pd.Timestamp('2024-06-24')
start_date = end_date - pd.DateOffset(days=599)

# 데이터와 파일명을 사전으로 정리
data_sources = {
    'KR3YT=RR': 'KR_Gov_3Y.csv',
    'US10YT=RR': 'US_Gov_10Y.csv',
    'USD/KRW': 'USD_KRW_Exchange_Rate.csv',
    'EUR/USD': 'EUR_USD_Exchange_Rate.csv',
    'GC=F': 'Gold_Prices.csv',
    'US500': 'SP500.csv',
    'IXIC': 'NASDAQ.csv',
    'DJI': 'DowJones.csv',
    'KQ11': 'KOSDAQ.csv',
    'KS11': 'KOSPI.csv',
     'DX-Y.NYB': 'Dollar_Index.csv'
}

# 각 데이터 소스에 대해 데이터를 읽고, CSV로 저장
for symbol, filename in data_sources.items():
    data = fdr.DataReader(symbol, start_date, end_date)
    data.to_csv(filename)
    print(f'Data for {symbol} saved to {filename}')
