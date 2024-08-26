import pandas as pd
import pandas_ta as ta

# 1. 데이터 로드 (CSV 대신 DataFrame 직접 생성)
stock_file = '005930_삼성전자_daily_data.csv'
# 종목 데이터 로드
df = pd.read_csv(stock_file, encoding='utf-8')

df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df.set_index('Date', inplace=True)

df = df.sort_values(by='Date')  # 날짜 순으로 정렬

# OHLCV 열에 0이 있는지 확인하는 코드
columns_to_check = ['open', 'high', 'low', 'close', 'volume']

# 각 열에서 0이 있는지 확인하고 그 결과를 출력
for column in columns_to_check:
    zero_rows = df[df[column] == 0]  # 0이 있는 행을 필터링
    if not zero_rows.empty:
        print(f"'{column}' 열에 0이 존재하는 행들:")
        print(zero_rows[[column]])  # 해당 열의 0이 있는 행과 날짜를 출력

        # 0을 NaN으로 대체
        df.replace({column: 0}, pd.NA, inplace=True)  # inplace로 원본 수정

        df.bfill(inplace=True)  # 이전 날짜 값으로 채우기
        df.ffill(inplace=True)  # 이후 날짜 값으로 채우기

        # 수정된 행만 필터링하여 출력 (원래 0이 있던 위치에서)
        modified_rows = df.loc[zero_rows.index, [column]]
        print(f"'{column}' 열에서 수정된 행들:")
        print(modified_rows)
    else:
        print(f"'{column}' 열에 0이 없습니다.")


# 2. 기술적 지표 계산
# 트렌드 지표
df['SMA'] = ta.sma(df['close'], length=20)
df['EMA'] = ta.ema(df['close'], length=20)
df['WMA'] = ta.wma(df['close'], length=20)
macd = ta.macd(df['close'])
df['MACD'] = macd['MACD_12_26_9']
df['MACD_hist'] = macd['MACDh_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
# ADX 계산 (여러 열 반환)
adx = ta.adx(df['high'], df['low'], df['close'])
df['ADX'] = adx['ADX_14']
df['DMP'] = adx['DMP_14']  # +DI
df['DMN'] = adx['DMN_14']  # -DI
psar = ta.psar(df['high'], df['low'], df['close'])
df['PSAR_up'] = psar['PSARl_0.02_0.2']  # 상승 추세일 때의 PSAR 값
df['PSAR_down'] = psar['PSARs_0.02_0.2']  # 하락 추세일 때의 PSAR 값

# 모멘텀 지표
df['RSI'] = ta.rsi(df['close'], length=20)
stoch = ta.stoch(df['high'], df['low'], df['close'])
df['Stoch_D'] = stoch['STOCHd_14_3_3']
df['Stoch_K'] = stoch['STOCHk_14_3_3']
df['CCI'] = ta.cci(df['high'], df['low'], df['close'])
df['Williams %R'] = ta.willr(df['high'], df['low'], df['close'])
df['ROC'] = ta.roc(df['close'], length=20)
df['Momentum'] = ta.mom(df['close'], length=20)

# 변동성 지표
bbands = ta.bbands(df['close'], length=20, std=2.0)
df['BB_upper'], df['BB_middle'], df['BB_lower'] = bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=20)
keltner = ta.kc(df['close'], df['high'], df['low']) # @TODO
df['Keltner_channel_upper'] = keltner['KCLe_20_2']
df['Keltner_channel_middle'] = keltner['KCBe_20_2']
df['Keltner_channel_lower'] = keltner['KCLe_20_2']

# 거래량 지표
df['OBV'] = ta.obv(df['close'], df['volume'])
df['A/D Line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
df['VROC'] = ta.roc(df['volume'], length=20)

# 결합 지표
df['MACD_Hist'] = df['MACD_hist']  # 이미 MACD에서 계산됨
df['CMO'] = ta.cmo(df['close'], length=20)
trix = ta.trix(df['close'], length=20)
df['TRIX'] = trix['TRIX_20_9']
df['TRIX_signal'] = trix['TRIXs_20_9']

# 기타 지표
vortex = ta.vortex(df['high'], df['low'], df['close'], 20)
df['Vortex_Plus'] = vortex['VTXP_20']
df['Vortex_Minus'] = vortex['VTXM_20']
# Aroon 지표 계산 (여러 열 반환)
aroon = ta.aroon(df['high'], df['low'], length=20)

# 반환된 DataFrame에서 필요한 열을 선택하여 할당
df['Aroon_Up'] = aroon['AROONU_20']
df['Aroon_Down'] = aroon['AROOND_20']
df['Ultimate_Oscillator'] = ta.uo(df['high'], df['low'], df['close'])

# 3. 결과를 CSV 파일로 저장
df.to_csv(f'ti_{stock_file}')

# 4. 출력 데이터프레임 확인 (선택적)
print(df.head())
