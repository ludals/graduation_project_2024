# pip install pykiwoom

from pykiwoom.kiwoom import *

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

# 연결상태 확인
state = kiwoom.GetConnectState()
if state == 0:
    print("미연결")
elif state == 1:
    print("연결완료")


# 사용자 정보 가져오기
account_num = kiwoom.GetLoginInfo("ACCOUNT_CNT")        # 전체 계좌수
accounts = kiwoom.GetLoginInfo("ACCNO")                 # 전체 계좌 리스트
user_id = kiwoom.GetLoginInfo("USER_ID")                # 사용자 ID
user_name = kiwoom.GetLoginInfo("USER_NAME")            # 사용자명
keyboard = kiwoom.GetLoginInfo("KEY_BSECGB")            # 키보드보안 해지여부
firewall = kiwoom.GetLoginInfo("FIREW_SECGB")           # 방화벽 설정 여부

print("***** user_information *****")
print(f"account_num: {account_num}")
print(f"accounts: {accounts}")
print(f"user_id: {user_id}")
print(f"user_name: {user_name}")
print(f"keyboard: {keyboard}")
print(f"firewall: {firewall}")

# 종목 코드 얻기
kospi = kiwoom.GetCodeListByMarket('0')
kosdaq = kiwoom.GetCodeListByMarket('10')
etf = kiwoom.GetCodeListByMarket('8')

print(len(kospi), kospi)
print(len(kosdaq), kosdaq)
print(len(etf), etf)

# 종목명 얻기
name = kiwoom.GetMasterCodeName("005930")
print(name)
