export const predefinedSizes: { [key: string]: number } =  {
  "HD 한국조선해양": 13574300000000,  // 약 13.57조 원
  "HD 현대일렉트릭": 11320000000000,  // 약 11.32조 원
  "HD 현대중공업": 13430000000000,  // 약 13.43조 원
  "HMM": 13210000000000,  // 약 13.21조 원
  "KB 금융": 35020000000000,  // 약 35.02조 원
  "KT": 9980000000000,  // 약 9.98조 원
  "LG": 12780000000000,  // 약 12.78조 원
  "LG 에너지솔루션": 84010000000000,  // 약 84.01조 원
  "LG 이노텍": 6010000000000,  // 약 6.01조 원
  "LG 전자": 16950000000000,  // 약 16.95조 원
  "LG 화학": 23880000000000,  // 약 23.88조 원
  "LIG 넥스원": 4160000000000,  // 약 4.16조 원
  "LS": 3720000000000,  // 약 3.72조 원
  "LS ELECTRIC": 5110000000000,  // 약 5.11조 원
  "LS 에코에너지": 8529030000000,  // 약 8.53조 원
  "NAVER": 26510000000000,  // 약 26.51조 원
  "POSCO 홀딩스": 28750000000000,  // 약 28.75조 원
  "SK": 10350000000000,  // 약 10.35조 원
  "SKC": 4740000000000,  // 약 4.74조 원
  "SK 스퀘어": 11000000000000,  // 약 11.00조 원
  "SK 이노베이션": 10190000000000,  // 약 10.19조 원
  "SK 텔레콤": 12050000000000,  // 약 12.05조 원
  "SK 하이닉스": 135040000000000,  // 약 135.04조 원
  "고려아연": 11060000000000,  // 약 11.06조 원
  "금양": 3200000000000,  // 약 3.20조 원
  "기아": 41590000000000,  // 약 41.59조 원
  "기업은행": 11290000000000,  // 약 11.29조 원
  "대원전선": 2575250000000,  // 약 2.58조 원
  "대한전선": 23800000000000,  // 약 2.38조 원
  "두산에너빌리티": 11540000000000,  // 약 11.54조 원
  "메리츠 금융지주": 17930000000000,  // 약 17.93조 원
  "삼성 SDI": 23070000000000,  // 약 23.07조 원
  "삼성물산": 26780000000000,  // 약 26.78조 원
  "삼성바이오로직스": 68110000000000,  // 약 68.11조 원
  "삼성생명": 19480000000000,  // 약 19.48조 원
  "삼성에스디에스": 11520000000000,  // 약 11.52조 원
  "삼성전기": 10880000000000,  // 약 10.88조 원
  "삼성전자": 515210000000000,  // 약 515.21조 원
  "삼성전자우": 517595730000000,  // 약 517.60조 원
  "삼성중공업": 9290000000000,  // 약 9.29조 원
  "삼성화재": 17740000000000,  // 약 17.74조 원
  "삼양식품": 3770000000000,  // 약 3.77조 원
  "셀트리온": 43950000000000,  // 약 43.95조 원
  "신한지주": 30105100000000,  // 약 30.10조 원
  "아모레퍼시픽": 7510000000000,  // 약 7.51조 원
  "영풍제지": 6256500000000,  // 약 6.26조 원
  "우리금융지주": 12280000000000,  // 약 12.28조 원
  "유한양행": 8680000000000,  // 약 8.68조 원
  "이수스페셜티케미컬": 1370000000000,  // 약 1.37조 원
  "이수페타시스": 2830000000000,  // 약 2.83조 원
  "카카오": 16610000000000,  // 약 16.61조 원
  "카카오뱅크": 10440000000000,  // 약 10.44조 원
  "크래프톤": 16670000000000,  // 약 16.67조 원
  "포스코인터내셔널": 9320000000000,  // 약 9.32조 원
  "포스코퓨처엠": 16540000000000,  // 약 16.54조 원
  "하나금융지주": 19680000000000,  // 약 19.68조 원
  "하이브": 7360000000000,  // 약 7.36조 원
  "한국가스공사": 4490000000000,  // 약 4.49조 원
  "한국석유": 13610000000000,  // 약 13.61조 원
  "한국전력": 13610000000000,  // 약 13.61조 원
  "한미반도체": 11430000000000,  // 약 11.43조 원
  "한화솔루션": 4460000000000,  // 약 4.46조 원
  "한화에어로스페이스": 13950000000000,  // 약 13.95조 원
  "한화오션": 10160000000000,  // 약 10.16조 원
  "현대로템": 5880000000000,  // 약 5.88조 원
  "현대모비스": 20410000000000,  // 약 20.41조 원
  "현대차": 63300000000000,  // 약 63.3조 원
  "에코프로비엠": 17020000000000,  // 약 17.02조 원
  "알테오젠": 16740000000000,  // 약 16.74조 원
  "KT&G": 13980000000000,  // 약 13.98조 원
  "에코프로": 11560000000000,  // 약 11.56조 원
  "HLB": 11230000000000  // 약 11.23조 원
};

function calculateProportionalSizes(predefinedSizes: { [key: string]: number }): { [key: string]: number } {
  const totalMarketCap = Object.values(predefinedSizes).reduce((sum: number, marketCap: number) => sum + marketCap, 0);

  const proportionalSizes: { [key: string]: number } = {};
  for (const [company, marketCap] of Object.entries(predefinedSizes)) {
    proportionalSizes[company] = (marketCap / totalMarketCap) * 1000;
  }
  return proportionalSizes;
}




export const proportionalSizes = calculateProportionalSizes(predefinedSizes);
