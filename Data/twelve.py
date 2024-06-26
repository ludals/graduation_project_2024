import requests

API_KEY = '187d5deb48ef4e87a9653c7f166eb092'

symbol = '005930'

# twelvedata API URL 설정
url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&apikey={API_KEY}'

# API 호출
response = requests.get(url)
data = response.json()

# 주가 출력
if 'values' in data:
    # 가장 최근 데이터 가져오기
    latest_data = data['values'][0]
    price = latest_data['close']
    print(f"현재 삼성전자 주가는 {price}원 입니다.")
else:
    print("주가를 가져오는 데 실패했습니다. 오류 메시지:", data)
