from pykiwoom.kiwoom import *
import pandas as pd
import time

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

# 5분봉 데이터 수집을 위한 설정
code = "005930"  # 삼성전자
df_5min = pd.DataFrame()

next_request = 0
while True:
    try:
        # 5분봉 데이터 요청
        df = kiwoom.block_request("opt10080",
                                  종목코드=code,
                                  틱범위="60",  # 5분봉
                                  output="주식분봉차트조회",
                                  next=next_request)

        df_5min = pd.concat([df_5min, pd.DataFrame(df)], ignore_index=True)

        # 다음 데이터가 있는지 확인
        next_count = kiwoom.GetRepeatCnt("opt10080", "주식분봉차트조회")
        if next_count == 0:
            break

        next_request += 1
        time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기

    except Exception as e:
        print(f"Error fetching data for stock code {code}: {e}")
        time.sleep(5)  # 에러 발생 시 잠시 대기 후 재시도
        break  # 영구 루프 방지를 위해 에러 발생 시 반복 중단

# CSV 파일로 저장
df_5min.to_csv("005930_60min_data.csv", index=False)
print("5분봉 데이터 CSV 추출 완료")
