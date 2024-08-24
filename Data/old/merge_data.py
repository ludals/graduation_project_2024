import pandas as pd

# 각 파일의 경로를 지정합니다.
file1 = 'kospi_kosdaq_data.csv'
file2 = 'krw_based_fx_commodities.csv'
file3 = 'market_data_1.csv'

# 첫 번째 열을 'Date'로 설정하여 각 파일을 읽어옵니다.
df1 = pd.read_csv(file1, header=0, index_col=0)  # 첫 번째 행을 열 이름으로, 첫 번째 열을 인덱스로 읽어오기
df1.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
df1.reset_index(inplace=True)  # 인덱스를 열로 변환
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')  # 날짜 형식으로 변환, 오류 발생 시 NaT로 처리

df2 = pd.read_csv(file2, header=0, index_col=0)  # 첫 번째 행을 열 이름으로, 첫 번째 열을 인덱스로 읽어오기
df2.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
df2.reset_index(inplace=True)  # 인덱스를 열로 변환
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')  # 날짜 형식으로 변환, 오류 발생 시 NaT로 처리

df3 = pd.read_csv(file3, header=0, index_col=0)  # 첫 번째 행을 열 이름으로, 첫 번째 열을 인덱스로 읽어오기
df3.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
df3.reset_index(inplace=True)  # 인덱스를 열로 변환
df3['Date'] = pd.to_datetime(df3['Date'], errors='coerce')  # 날짜 형식으로 변환, 오류 발생 시 NaT로 처리

# NaT(날짜 형식 오류) 제거
df1.dropna(subset=['Date'], inplace=True)
df2.dropna(subset=['Date'], inplace=True)
df3.dropna(subset=['Date'], inplace=True)

# 세 데이터프레임을 'Date' 열을 기준으로 병합합니다.
merged_df = df1.merge(df2, on='Date', how='inner').merge(df3, on='Date', how='inner')

# 날짜 필터링 (2003-12-01부터 2024-08-22까지)
start_date = '2003-12-01'
end_date = '2024-08-22'
filtered_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]

# 결과를 새로운 CSV 파일로 저장합니다.
filtered_df.to_csv('merged_filtered_data.csv', index=False, encoding='utf-8-sig')

print("Data saved to 'merged_filtered_data.csv'.")
