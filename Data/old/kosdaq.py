import investpy
import pandas as pd

# KOSDAQ 데이터 다운로드
kosdaq_df = investpy.get_index_historical_data(index='KOSDAQ', country='south korea', from_date='01/01/2022', to_date='22/08/2024')

# CSV 파일로 저장
kosdaq_df.to_csv('kosdaq_investing_data.csv', encoding='utf-8-sig')
print("KOSDAQ data saved to 'kosdaq_investing_data.csv'.")
