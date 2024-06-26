from pykiwoom.kiwoom import *
import pandas as pd
import time

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

# 종목 리스트
stock_list = [
    "005930", "047050", "000660", "036460", "096770", "001530", "068270", "006340",
    "032830", "005380", "078930", "000270", "267260", "105560", "272210", "010120",
    "314130", "000990", "103140", "012450", "042700", "241560", "001440", "035420",
    "247540", "259960", "005935", "034020", "006260", "042660", "373220", "207940",
    "005490", "006400", "051910", "055550", "028260", "012330", "003670", "035720",
    "066570", "086790", "000810", "032830", "138040", "011200", "402340", "329180",
    "003550", "015760", "018260", "034730", "033780", "017670", "009150", "024110",
    "009540", "316140", "010130", "323410", "090430", "030200", "352820", "086280"
]

# 데이터프레임 리스트 초기화
df_list = []

# 각 종목에 대해 데이터 수집
for code in stock_list:
    print(f"Fetching data for stock code: {code}")
    df_temp = pd.DataFrame()
    next_request = 0

    while True:
        try:
            df = kiwoom.block_request("opt10081",
                                      종목코드=code,
                                      기준일자="20240624",  # 필요에 따라 기준일자 변경
                                      수정주가구분=1,
                                      output="주식일봉차트조회",
                                      next=next_request)

            df_temp = pd.concat([df_temp, pd.DataFrame(df)], ignore_index=True)

            # 다음 데이터가 있는지 확인
            next_count = kiwoom.GetRepeatCnt("opt10081", "주식일봉차트조회")
            if next_count == 0:
                break

            next_request += 1
            time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기

        except Exception as e:
            print(f"Error fetching data for stock code {code}: {e}")
            time.sleep(5)  # 에러 발생 시 잠시 대기 후 재시도

    df_list.append(df_temp)
    time.sleep(1)  # 종목 간 대기 시간 추가

# 모든 종목 데이터프레임을 하나의 데이터프레임으로 병합
result_df = pd.concat(df_list, keys=stock_list)

# 변경할 컬럼명을 사전으로 매핑
column_mapping = {
    '종목코드': 'Ticker',
    '일자': 'DateTime',
    '현재가': 'Close',
    '시가': 'Open',
    '저가': 'Low',
    '고가': 'High',
    '거래량': 'Volume',
}

# 매핑된 컬럼명들만을 포함하는 새로운 데이터프레임 생성
selected_columns = list(column_mapping.keys())
filtered_df = result_df[selected_columns]

# 기존 컬럼 순서에 맞춰 새로운 컬럼명 리스트 생성
new_columns = [column_mapping[col] for col in selected_columns]

# 컬럼명 변경
filtered_df.columns = new_columns

# 엑셀 파일로 저장
filtered_df.to_csv("stock_daily_data_new_col.csv", index=False)

print("데이터 추출 완료")
