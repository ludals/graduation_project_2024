import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_exchange_rates(base_currency, symbols, start_date, end_date):
    api_url = "https://api.exchangeratesapi.io/history"
    payload = {
        "start_at": start_date.strftime('%Y-%m-%d'),
        "end_at": end_date.strftime('%Y-%m-%d'),
        "base": base_currency,
        "symbols": symbols
    }
    response = requests.get(api_url, params=payload)
    data = response.json()

    # API 응답을 출력하여 확인
    print("API Response:", data)

    # 데이터 추출을 시도하기 전에 'rates' 키 존재 여부 확인
    if 'rates' in data:
        return pd.DataFrame({
            "date": list(data['rates'].keys()),
            "rate": [list(day.values())[0] for day in data['rates'].values()]
        })
    else:
        return pd.DataFrame()  # 비어있는 데이터프레임 반환

# 기준일자 설정 (예: 최근 30일)
end_date = datetime.now()
start_date = end_date - timedelta(days=600)

# 환율 데이터 수집 (예: 유로 대 미국달러)
df_exchange = fetch_exchange_rates("EUR", "USD", start_date, end_date)

# 데이터프레임 유효성 검사
if not df_exchange.empty:
    # CSV 파일로 저장
    df_exchange.to_csv("EUR_USD_exchange_rate_data.csv", index=False)
    print("환율 데이터 일봉 CSV 추출 완료")
else:
    print("No data to save or API response was invalid.")
