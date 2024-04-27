import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import FinanceDataReader as fdr

# 데이터 로드
df = fdr.DataReader('KS11', '2020')

# 날짜 데이터에서 연도만 추출하여 'year' 컬럼 생성
df['year'] = df.index.year

# 'diff' 컬럼 생성 (예: 일일 종가의 차이)
df['diff'] = df['Close'].diff()

# 폰트 설정
font_location = ''
if platform.system() == 'Windows':
    font_location = 'C:\\USERS\\RLARU\\APPDATA\\LOCAL\\MICROSOFT\\WINDOWS\\FONTS\\NANUMGOTHIC.TTF'
  # 경로 확인 필요
elif platform.system() == 'Linux':
    font_location = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if font_location:
    font_name = fm.FontProperties(fname=font_location).get_name()
    print(font_name)
    matplotlib.rc('font', family='NanumGothic')

matplotlib.rc('axes', unicode_minus=False)
# 시각화
fig, ax = plt.subplots(figsize=(15, 5))

# 선 그래프로 'Close' 가격 표시
ax.plot(df.index, df['Close'], label='코스피 종가', color='blue', marker='o', linestyle='-')

# 선 그래프로 'diff' 표시 (일일 종가 변화)
ax.plot(df.index, df['diff'].fillna(0), label='일일 종가 차이', color='red', linestyle='--')

ax.set_title('코스피 지수 (KS11) 종가 및 일일 변화')
ax.set_xlabel('날짜')
ax.set_ylabel('종가 (포인트)')
ax.legend()

plt.xticks(rotation=45)  # x축 레이블 회전
plt.tight_layout()  # 레이아웃 조정
plt.show()
