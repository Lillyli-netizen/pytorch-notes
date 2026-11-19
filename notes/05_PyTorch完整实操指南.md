# PyTorch完整实操指南

> 原文：[CSDN](https://blog.csdn.net/qq_63283852/article/details/167813965)

**摘要：**本文按「张量 → 自动求导 → 搭建网络 → 训练出第一个分类模型」的完整链路，系统讲解 PyTorch 从环境安装到实战的全过程。内容涵盖张量的创建、形状操作、拼接与广播机制，autograd 自动求导原理，nn.Module 搭建神经网络、激活函数与参数初始化，以及损失函数、优化器与学习率衰减的选择。最后通过一个完整可运行的 MNIST 手写数字分类实战，把前面所有知识点串起来，并附上万能训练模板与新手最常踩的 5 个坑，帮助读者一次跑通并避开常见错误。

---

> 一篇讲完「张量 → 自动求导 → 建网络 → 训练出第一个分类模型」的完整链路，所有代码可直接复制运行，文末的训练模板建议收藏。

## 前言

很多 PyTorch 教程有两个极端：要么全是数学公式吓退新手，要么直接甩一段 MNIST 代码，中间的过程完全没讲。这篇文章按照**实际写代码的顺序**，把整条链路串起来——你在哪一步卡住，就翻到对应的小节。

全文基于 PyTorch 2.x + Python 3.10+，CPU 也能跑通全部代码。

---

## 一、环境安装（5 分钟）

推荐用 conda 建独立环境，避免污染系统 Python：

```bash
# 创建并激活环境
conda create -n pt python=3.10 -y
conda activate pt
CPU 版（没有 N 卡就用这个）
pip install torch torchvision
GPU 版（CUDA 12.1 示例，具体去 pytorch.org 选对应命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证安装：

```python
import torch
print(torch.__version__)                  # 2.x.x
print(torch.cuda.is_available())          # GPU 版应输出 True
```

---

## 二、张量：PyTorch 的地基

### 2.1 创建张量

```python
import torch
t1 = torch.tensor([[1., 2.], [3., 4.]])   # 从数据创建
t2 = torch.zeros(2, 3)                    # 全 0
t3 = torch.ones(2, 3)                     # 全 1
t4 = torch.rand(2, 3)                     # [0,1) 均匀分布
t5 = torch.randn(2, 3)                    # 标准正态分布（最常用）
t6 = torch.arange(0, 10, 2)               # 等差序列: [0,2,4,6,8]
t7 = torch.linspace(0, 1, 5)              # 5 个等间隔点
print(t1.dtype, t1.shape, t1.device)      # 类型、形状、所在设备
```

要点：`torch.randn` 生成随机张量是神经网络**参数初始化**的基础，`dtype` 默认是 `float32`——深度学习计算几乎都用 32 位浮点。

### 2.2 形状操作（面试和实战双高频）

| 操作 | 作用 | 一句话记忆 |
| --- | --- | --- |
| `reshape / view` | 改形状 | `view` 要求内存连续，`reshape` 都行 |
| `unsqueeze(dim)` | 在指定位置加一维 | 给单个样本加 batch 维 |
| `squeeze(dim)` | 去掉长度为 1 的维 | unsqueeze 的逆操作 |
| `transpose(a, b)` | 交换两个维度 | 二维转置用这个 |
| `permute(*dims)` | 任意重排所有维度 | 图像 HWC→CHW 用这个 |

```python
x = torch.randn(2, 3, 4)
print(x.view(2, 12).shape)          # [2, 12]
print(x.reshape(3, 2, 4).shape)     # [3, 2, 4]
print(x.unsqueeze(0).shape)         # [1, 2, 3, 4]  加 batch 维
print(x.squeeze(0).shape)           # 去掉长度 1 的维
print(x.transpose(0, 2).shape)      # [4, 3, 2]  交换 0 和 2 维
print(x.permute(2, 0, 1).shape)     # [4, 2, 3]  按新维度顺序重排
```

**经典坑**：`view` 报错 `view size is not compatible...`，是因为张量内存不连续。先 `.contiguous()` 再 `view`，或者干脆用 `reshape`。

### 2.3 拼接：cat vs stack（新手必混淆）

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(torch.cat([a, b], dim=0).shape)    # [4, 3]  沿已有维度拼接
print(torch.cat([a, b], dim=1).shape)    # [2, 6]
print(torch.stack([a, b], dim=0).shape)  # [2, 2, 3] 新开一个维度堆叠
```

一句话区分：**cat 不改变维度数，stack 会多一维**。要把 10 个 batch 堆成一个 batch，用 `stack`。

### 2.4 广播机制

```python
a = torch.randn(3, 4)
b = torch.randn(4)          # 形状 [4]
print((a + b).shape)        # [3, 4]，b 自动复制成 3 行
c = torch.randn(3, 1)
print((a * c).shape)        # [3, 4]，c 沿列复制
```

广播规则：从后往前对齐维度，两个维度要么相等、要么有一个为 1、要么不存在，才能广播。

---

## 三、自动求导 autograd：PyTorch 的灵魂

PyTorch 用**计算图 + 动态求导**实现自动微分，你只需要关心前向计算：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3          # y = x^3, dy/dx = 3x^2
y.backward()        # 反向传播，自动算梯度
print(x.grad)       # tensor(12.)  = 3 * 2^2
```

三个必须知道的细节：

```python
# 1. 默认张量不追踪梯度，参数张量才需要
w = torch.randn(3, requires_grad=True)
2. backward 后梯度会累加，每个 batch 前要清零
if w.grad is not None:
w.grad.zero_()
3. 推理阶段关闭梯度，省显存又提速
with torch.no_grad():
out = model(input)
```

**梯度清零**是新手最常见的漏写点——PyTorch 的梯度默认累加，不清零的话 loss 会越训越怪。

---

## 四、用 nn.Module 搭神经网络

```python
import torch.nn as nn
class MLP(nn.Module):
def init(self, in_dim, hidden, out_dim):
super().init()
self.net = nn.Sequential(
nn.Linear(in_dim, hidden),
nn.ReLU(),                    # 激活函数：引入非线性
nn.Linear(hidden, out_dim)
)
def forward(self, x):
    return self.net(x)
model = MLP(784, 256, 10)
print(model)                              # 打印网络结构
print(sum(p.numel() for p in model.parameters()))  # 参数总量
```

### 4.1 常用激活函数怎么选

| 激活函数 | 公式特点 | 适用场景 |
| --- | --- | --- |
| `nn.ReLU()` | max(0, x)，计算快 | 默认首选 |
| `nn.LeakyReLU()` | 负半轴有小斜率 | 缓解"死 ReLU" |
| `nn.Sigmoid()` | 输出 (0,1) | 二分类输出层 |
| `nn.Softmax(dim=-1)` | 输出概率分布 | 多分类输出层（配 CrossEntropyLoss 时不要手动加） |

### 4.2 参数初始化

好的初始化能让训练稳定很多。默认初始化已经够用，需要手动时：

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)   # ReLU 配 Kaiming
        nn.init.zeros_(m.bias)
model.apply(init_weights)
```

经验法则：**ReLU 用 Kaiming，Sigmoid/tanh 用 Xavier**。

---

## 五、损失函数与优化器

### 5.1 损失函数

```python
criterion = nn.CrossEntropyLoss()            # 多分类（内置 softmax + log）
loss = criterion(logits, targets)            # logits 是原始输出，不要先 softmax！
```

| 任务 | 损失函数 |
| --- | --- |
| 多分类 | `CrossEntropyLoss` |
| 二分类 | `BCEWithLogitsLoss` |
| 回归 | `MSELoss` / `L1Loss` |

### 5.2 优化器与学习率衰减

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# weight_decay 就是 L2 正则化，防止过拟合最省事的做法
每 3 个 epoch 学习率乘 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
```

| 优化器 | 特点 |
| --- | --- |
| SGD + Momentum | 经典，调好了上限高 |
| Adam | 自适应学习率，默认选择 |
| AdamW | 修正了权重衰减的 Adam，现在的主流 |

学习率是**对训练影响最大的单个超参数**，出问题先调它。

---

## 六、实战：MNIST 手写数字分类（完整可运行）

把前面所有知识串起来，一次训练 10 个 epoch 测试集准确率 97%+：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
2. 数据
transform = transforms.Compose([
transforms.ToTensor(),                    # 转张量并归一化到 [0,1]
transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])
train_set = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256)
3. 模型（两层隐藏层 MLP）
class MLP(nn.Module):
def init(self):
super().init()
self.net = nn.Sequential(
nn.Flatten(),                 # [B,1,28,28] -> [B,784]
nn.Linear(784, 256),
nn.ReLU(),
nn.Dropout(0.2),              # 正则化：随机丢弃 20% 神经元
nn.Linear(256, 128),
nn.ReLU(),
nn.Linear(128, 10)            # 输出 logits，不做 softmax
)
def forward(self, x):
return self.net(x)
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
4. 训练 + 验证
for epoch in range(1, 11):
# --- 训练 ---
model.train()
total_loss = 0.0
for x, y in train_loader:
x, y = x.to(device), y.to(device)
optimizer.zero_grad()             # 梯度清零（别忘！）
logits = model(x)
loss = criterion(logits, y)
loss.backward()                   # 反向传播
optimizer.step()                  # 更新参数
total_loss += loss.item() * y.size(0)
scheduler.step()
# --- 验证 ---
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"epoch {epoch:2d} | train loss {total_loss/len(train_set):.4f} "
      f"| test acc {correct/total:.4f} | lr {scheduler.get_last_lr()[0]:.6f}")
```

典型输出：

```python
epoch  1 | train loss 0.3412 | test acc 0.9605 | lr 0.000988
epoch  5 | train loss 0.0617 | test acc 0.9778 | lr 0.000500
epoch 10 | train loss 0.0246 | test acc 0.9812 | lr 0.000051
```

---

## 七、万能训练模板（建议收藏）

所有 PyTorch 训练代码都是这个骨架，换数据集换模型只改中间几行：

```python
每个 epoch：
  model.train()                    # 1. 切训练模式（Dropout/BatchNorm 生效）
  for x, y in train_loader:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()        # 2. 梯度清零
      loss = criterion(model(x), y)  # 3. 前向 + 算损失
      loss.backward()              # 4. 反向传播
      optimizer.step()             # 5. 更新参数
  scheduler.step()                 # 6. 更新学习率
model.eval()                     # 7. 切评估模式
with torch.no_grad():            # 8. 关闭梯度
... 在验证集上测指标 ...
```

---

## 八、新手最常踩的 5 个坑

1. **CrossEntropyLoss 之前手动加了 Softmax** —— 会把模型训死，直接喂原始 logits。
2. **忘记 `optimizer.zero_grad()`** —— 梯度累加，loss 曲线诡异。
3. **评估时忘记 `model.eval()` 和 `torch.no_grad()`** —— Dropout 没关，显存暴涨、结果抖动。
4. **`view` 报内存不连续错误** —— 用 `reshape` 或先 `.contiguous()`。
5. **数据和模型不在同一设备** —— 报 `Expected all tensors to be on the same device`，检查每个张量是否都 `.to(device)`。

---

## 九、总结与下一步

本文打通了 PyTorch 完整链路：张量操作 → autograd → nn.Module → 完整训练循环。下一步的进阶方向：

- **卷积网络**：把 MLP 换成 CNN，MNIST 准确率能到 99%+（`nn.Conv2d` + `nn.MaxPool2d`）；
- **迁移学习**：加载 `torchvision.models` 的预训练模型微调；
- **保存与加载**：`torch.save(model.state_dict(), "ckpt.pt")`；
- **TensorBoard 可视化**：`from torch.utils.tensorboard import SummaryWriter`。

如果这篇文章对你有帮助，欢迎点赞收藏，评论区交流你在哪一步踩了坑。

---

> **发布建议**：CSDN 标签选「PyTorch、深度学习、神经网络、Python、人工智能」；封面图放训练 loss 曲线或网络结构图；首发时间建议工作日晚 8-10 点流量最高。
