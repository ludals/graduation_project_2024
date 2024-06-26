from pykiwoom.kiwoom import *
import pandas as pd
import time

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

# KOSPI 지수 코드 설정
kospi_index_code = "001"

# 데이터프레임 초기화
df_kospi = pd.DataFrame()

next_request = 0

# 지수 데이터 수집
while True:
    try:
        df = kiwoom.block_request("opt20003",
                                  업종코드=kospi_index_code,
                                  output="전업종지수",
                                  next=next_request)

        df_kospi = pd.concat([df_kospi, pd.DataFrame(df)], ignore_index=True)

        # 다음 데이터가 있는지 확인
        next_count = kiwoom.GetRepeatCnt("opt20003", "전업종지수")
        if next_count == 0:
            break

        next_request += 1
        time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기

    except Exception as e:
        print(f"Error fetching KOSPI index data: {e}")
        time.sleep(5)  # 에러 발생 시 잠시 대기 후 재시도
        break

# 데이터 확인
print("데이터프레임 헤드:")
print(df_kospi.head())
print("데이터프레임 정보:")
print(df_kospi.info())

# CSV 파일로 저장
df_kospi.to_csv("kospi_daily_data.csv", index=False)

print("KOSPI 지수 데이터 추출 완료")
