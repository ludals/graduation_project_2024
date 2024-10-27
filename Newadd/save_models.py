import torch
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 생성 및 학습 (예제 데이터 사용)
model = DLinear()
dummy_input = torch.randn(100, 10)  # 임시 학습 데이터
dummy_output = torch.randn(100, 1)  # 임시 레이블

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 간단한 학습 루프
for epoch in range(100):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_output)
    loss.backward()
    optimizer.step()

# 모델 저장
#torch.save(model, './models/AAPL.pth')
#torch.save(model, './models/TSLA.pth')
#torch.save(model, './models/GOOGL.pth')
