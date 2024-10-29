import pandas as pd
import pandas_ta as ta

# 회사 리스트
companies = [
    "009540_HD한국조선해양_daily_data",
    "267260_HD현대일렉트릭_daily_data",
    "329180_HD현대중공업_daily_data",
    "011200_HMM_daily_data",
    "105560_KB금융_daily_data",
    "030200_KT_daily_data",
    "003550_LG_daily_data",
    "373220_LG에너지솔루션_daily_data",
    "011070_LG이노텍_daily_data",
    "066570_LG전자_daily_data",
    "051910_LG화학_daily_data",
    "079550_LIG넥스원_daily_data",
    "006260_LS_daily_data",
    "010120_LSELECTRIC_daily_data",
    "229640_LS에코에너지_daily_data",
    "035420_NAVER_daily_data",
    "005490_POSCO홀딩스_daily_data",
    "034730_SK_daily_data",
    "011790_SKC_daily_data",
    "402340_SK스퀘어_daily_data",
    "096770_SK이노베이션_daily_data",
    "017670_SK텔레콤_daily_data",
    "000660_SK하이닉스_daily_data",
    "010130_고려아연_daily_data",
    "000270_기아_daily_data",
    "024110_기업은행_daily_data",
    "006340_대원전선_daily_data",
    "001440_대한전선_daily_data",
    "034020_두산에너빌리티_daily_data",
    "138040_메리츠금융지주_daily_data",
    "006400_삼성SDI_daily_data",
    "028260_삼성물산_daily_data",
    "207940_삼성바이오로직스_daily_data",
    "032830_삼성생명_daily_data",
    "018260_삼성에스디에스_daily_data",
    "009150_삼성전기_daily_data",
    "005930_삼성전자_daily_data",
    "005935_삼성전자우_daily_data",
    "010140_삼성중공업_daily_data",
    "000810_삼성화재_daily_data",
    "003230_삼양식품_daily_data",
    "068270_셀트리온_daily_data",
    "055550_신한지주_daily_data",
    "090430_아모레퍼시픽_daily_data",
    "006740_영풍제지_daily_data",
    "316140_우리금융지주_daily_data",
    "000100_유한양행_daily_data",
    "457190_이수스페셜티케미컬_daily_data",
    "007660_이수페타시스_daily_data",
    "035720_카카오_daily_data",
    "323410_카카오뱅크_daily_data",
    "259960_크래프톤_daily_data",
    "047050_포스코인터내셔널_daily_data",
    "003670_포스코퓨처엠_daily_data",
    "086790_하나금융지주_daily_data",
    "352820_하이브_daily_data",
    "036460_한국가스공사_daily_data",
    "004090_한국석유_daily_data",
    "015760_한국전력_daily_data",
    "042700_한미반도체_daily_data",
    "009830_한화솔루션_daily_data",
    "012450_한화에어로스페이스_daily_data",
    "042660_한화오션_daily_data",
    "064350_현대로템_daily_data",
    "012330_현대모비스_daily_data",
    "005380_현대차_daily_data",
    "247540_에코프로비엠_daily_data",
    "123123_알테오젠_daily_data",
    "033780_KT&G_daily_data",
    "086520_에코프로_daily_data",
    "028300_HLB_daily_data",
  ]


# 데이터를 저장할 빈 리스트
all_data = []

# 데이터 처리를 위한 루프
for company in companies:
    # CSV 파일 경로 구성 (예: '009540_HD한국조선해양_daily_data.csv')
    file_path = f'{company}.csv'
    
  # CSV 파일 읽기
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {company}")
    except FileNotFoundError:
        print(f"File not found for {company}, skipping.")
        continue

    # 첫 번째 열을 'Date'로 인식하고, 날짜 형식으로 변환
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)  # 첫 번째 열의 이름을 'Date'로 변경
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')  # 날짜 형식으로 변환

    # 날짜를 오름차순으로 정렬
    df.sort_values(by='Date', inplace=True)
    
    # 기술적 지표 계산 (예시)
    df['EMA_12'] = ta.ema(df['close'], length=12)
    df['EMA_26'] = ta.ema(df['close'], length=26)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
    
    # Bollinger Bands (BBANDS) 계산 및 출력
    bbands = ta.bbands(df['close'], length=20, std=2)
    if 'BBL' in bbands.columns:
        df['BBANDS_Lower'] = bbands['BBL']
        df['BBANDS_Middle'] = bbands['BBM']
        df['BBANDS_Upper'] = bbands['BBU']
    else:
        print(f"Unexpected Bollinger Bands columns for {company}, skipping Bollinger Bands.")
    
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    
    all_data.append(df)
    df.to_csv(f'processed_{company}.csv', index=False, encoding='utf-8-sig')



print("All data processed and saved.")