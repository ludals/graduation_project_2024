import yfinance as yf
import pandas as pd

# USD/KRW 환율 데이터 로드
usd_krw = yf.download('KRW=X', start='2000-01-01')  # USD/KRW

# 통화에 대한 USD 대비 환율 데이터 로드
usd_cny = yf.download('CNY=X', start='2000-01-01')  # USD/CNY
usd_jpy = yf.download('JPY=X', start='2000-01-01')  # USD/JPY
eur_usd = yf.download('EURUSD=X', start='2000-01-01')  # EUR/USD

# 금, 원유, 가스 데이터 로드 (달러 기준)
gold_usd = yf.download('GC=F', start='2000-01-01')  # 금
crude_oil_usd = yf.download('CL=F', start='2000-01-01')  # 원유
natural_gas_usd = yf.download('NG=F', start='2000-01-01')  # 천연가스

# 각 통화에 대해 원화 기준 환율을 계산
cny_krw = usd_krw['Close'] / usd_cny['Close']  # CNY/KRW = (USD/KRW) / (USD/CNY)
jpy_krw = usd_krw['Close'] / usd_jpy['Close']  # JPY/KRW = (USD/KRW) / (USD/JPY)
eur_krw = usd_krw['Close'] * eur_usd['Close']  # EUR/KRW = (USD/KRW) * (EUR/USD)

# 금, 원유, 가스 가격을 원화 기준으로 변환
gold_krw = gold_usd['Close'] * usd_krw['Close']  # 금 가격 (원화 기준)
crude_oil_krw = crude_oil_usd['Close'] * usd_krw['Close']  # 원유 가격 (원화 기준)
natural_gas_krw = natural_gas_usd['Close'] * usd_krw['Close']  # 천연가스 가격 (원화 기준)

# 데이터프레임 통합
combined_df = pd.concat({
    'USD/KRW': usd_krw['Close'],
    'CNY/KRW': cny_krw,
    'JPY/KRW': jpy_krw,
    'EUR/KRW': eur_krw,
    'Gold (KRW)': gold_krw,
    'Crude Oil (KRW)': crude_oil_krw,
    'Natural Gas (KRW)': natural_gas_krw
}, axis=1)

# 날짜별로 정렬
combined_df.sort_index(inplace=True)

# NaN 값 처리
combined_df.ffill(inplace=True)  # 앞의 값으로 NaN 채우기

# CSV 파일로 저장
combined_df.to_csv('krw_based_fx_commodities.csv', encoding='utf-8-sig')

print("Data saved to 'krw_based_fx_commodities.csv'.")
