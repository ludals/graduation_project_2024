import win32com.client
import pythoncom
import time

# 키움 API 초기화
kiwoom = win32com.client.Dispatch("KHOPENAPI.KHOpenAPICtrl.1")

# 이벤트 루프
class EventHandler:
    def OnEventConnect(self, errCode):
        if errCode == 0:
            print("로그인 성공")
        else:
            print("로그인 실패")

    def OnReceiveTrData(self, scrNo, rqName, trCode, recordName, prevNext, dataLen, errorCode, message, splmMsg):
        self.data = kiwoom.GetCommData(trCode, rqName, 0, "현재가")

# 로그인 함수
def login():
    event_handler = EventHandler()
    kiwoom.OnEventConnect.connect(event_handler.OnEventConnect)
    kiwoom.CommConnect()
    while kiwoom.GetConnectState() == 0:
        pythoncom.PumpWaitingMessages()
        time.sleep(0.1)

# 주식 실시간 데이터 요청 함수
def get_stock_data(stock_code):
    event_handler = EventHandler()
    kiwoom.OnReceiveTrData.connect(event_handler.OnReceiveTrData)
    kiwoom.SetInputValue("종목코드", stock_code)
    kiwoom.CommRqData("opt10001_req", "opt10001", 0, "0101")
    while not hasattr(event_handler, 'data'):
        pythoncom.PumpWaitingMessages()
        time.sleep(0.1)
    return event_handler.data.strip()

# 메인 함수
if __name__ == "__main__":
    login()
    stock_code = "005930"  # 삼성전자 종목 코드
    price = get_stock_data(stock_code)
    print(f"현재 삼성전자 주가는 {price}원 입니다.")
