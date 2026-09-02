# termux-train (AMEVA-Termux)

> **Native On-Device Deep Learning & LoRA Training Framework for Android Termux**  
> *Zero PyTorch Dependency · Pure Python Autograd Core · Pluggable NumPy Acceleration · Mobile-Resilient Runtime · On-Device LoRA · SafeTensors · RoPE Transformer*

---

## What is termux-train?

	ermux-train (also known as AMEVA-Termux) is a lightweight, self-contained deep learning and automatic differentiation (Autograd) training engine built specifically for **Android Termux native environments** and resource-constrained edge devices.

While standard mobile ML frameworks (TFLite, ONNX Runtime Mobile, ExecuTorch, NCNN) only support inference, 	ermux-train enables **full on-device training, backpropagation, RoPE Transformers, and LoRA fine-tuning** directly on smartphone hardware without requiring heavy PyTorch binaries or PRoot container virtualization.

---

## 5-Minute Quickstart

### 1. Installation

`ash
# In Android Termux:
pkg update && pkg install python python-numpy git
pip install termux-train
`

### 2. End-to-End On-Device Training (Python SDK)

`python
import termux_train as tt
import termux_train.nn as nn
import termux_train.optim as optim

# 1. Define Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 2. Synthetic Data
x_train = tt.randn(32, 4)
y_train = tt.randn(32, 1)

# 3. Training Loop with Autograd
for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")
`

---

## Official Documentation & Portal

- **Official Web Documentation**: [https://uno-km.vercel.app/lib/train/](https://uno-km.vercel.app/lib/train/)
- **GitHub Repository**: [https://github.com/uno-km/termux-train](https://github.com/uno-km/termux-train)
- **License**: Apache-2.0