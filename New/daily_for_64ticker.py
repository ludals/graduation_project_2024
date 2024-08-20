from pykiwoom.kiwoom import *
import pandas as pd
import time
import ticker

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

for key, value in ticker.ticker_dict.items():
    # 일봉 데이터 수집을 위한 설정
    code = key  # 종목코드
    df_daily = pd.DataFrame()
    if kiwoom.GetMasterCodeName(code) != "value":
        print("종목코드 불일치!")
        print(f"종목코드: {code}, dict_종목명: {value}, 종목검색결과: {kiwoom.GetMasterCodeName(code)}")
    print(f"Fetching daily data for {code} ({value})")
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    next_request = 0
    while True:
        try:
            # 일봉 데이터 요청
            df = kiwoom.block_request("opt10081",  # 일봉 데이터 요청
                                      종목코드=code,
                                      기준일자="20240819",  # 기준일자 설정 (필요에 따라 조정)
                                      수정주가구분="0",  # 수정주가 반영
                                      output="주식일봉차트조회",
                                      next=next_request)

            df_daily = pd.concat([df_daily, pd.DataFrame(df)], ignore_index=True)

            # 다음 데이터가 있는지 확인
            next_count = kiwoom.GetRepeatCnt("opt10081", "주식일봉차트조회")
            if next_count == 0:
                break

            next_request += 1
            time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기

        except Exception as e:
            print(f"Error fetching data for stock code {code}: {e}")
            time.sleep(5)  # 에러 발생 시 잠시 대기 후 재시도
            break  # 영구 루프 방지를 위해 에러 발생 시 반복 중단

    # 변경할 컬럼명을 사전으로 매핑
    column_mapping = {
        '일자': 'Date',
        '현재가': 'Close',
        '시가': 'Open',
        '저가': 'Low',
        '고가': 'High',
        '거래량': 'Volume',
    }

    # 매핑된 컬럼명들만을 포함하는 새로운 데이터프레임 생성
    selected_columns = list(column_mapping.keys())
    filtered_df = df_daily[selected_columns]

    # 기존 컬럼 순서에 맞춰 새로운 컬럼명 리스트 생성
    new_columns = [column_mapping[col] for col in selected_columns]

    # 컬럼명 변경
    filtered_df.columns = new_columns

    # DateTime 컬럼을 datetime 객체로 변환
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%Y%m%d')

    # 엑셀 파일로 저장
    filtered_df.to_csv(f"{code}_{kiwoom.GetMasterCodeName(code)}_daily_data.csv", index=False)

    time.sleep(61) # API 호출 제한을 피하기 위해 잠시 대기

print("일봉 데이터 CSV 추출 완료")
