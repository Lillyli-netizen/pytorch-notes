# PyTorch Notes

从深度学习基础到第一个图像分类网络的 PyTorch 学习笔记，按顺序编号。原文同步发布在 CSDN，本仓库保留转好的 Markdown 与原文快照。

作者：Olivia Li（[CSDN](https://blog.csdn.net/qq_63283852)）

## 目录

| # | 主题 | 笔记 | 原文 |
|---|---|---|---|
| 01 | 深度学习简介：ANN、张量、CNN/RNN、发展历史 | [notes/01](notes/01_深度学习简介.md) | [CSDN](https://blog.csdn.net/qq_63283852/article/details/166647747) |
| 02 | PyTorch 框架使用上：张量创建、类型、数值计算、索引 | [notes/02](notes/02_PyTorch框架使用上.md) | [CSDN](https://blog.csdn.net/qq_63283852/article/details/166850176) |
| 03 | PyTorch 框架进阶下：形状操作、拼接、autograd、线性回归 | [notes/03](notes/03_PyTorch框架进阶基础下.md) | [CSDN](https://blog.csdn.net/qq_63283852/article/details/166996034) |
| 04 | 神经网络基础：激活函数、参数初始化、搭建、损失函数 | [notes/04](notes/04_神经网络基础.md) | [CSDN](https://blog.csdn.net/qq_63283852/article/details/167498349) |
| 05 | PyTorch 完整实操指南：从张量到 MNIST 分类（附全码） | [notes/05](notes/05_PyTorch完整实操指南.md) | [CSDN](https://blog.csdn.net/qq_63283852/article/details/167813965) |

## 学习顺序

01 → 02 → 03 → 04 → 05。

- 01–02 解决「神经网络是什么」和「张量怎么造、怎么算」。
- 03 解决「形状怎么变、梯度怎么自动算」。
- 04 解决「网络怎么搭、损失怎么选」。
- 05 是一条完整可运行的主线，卡住时回查对应小节。

## 和其他仓库的关系

- [python-notes](https://github.com/Lillyli-netizen/python-notes)：Python 语法底子，读 02 之前至少过一遍函数、类、模块。
- [leetcode-notes](https://github.com/Lillyli-netizen/leetcode-notes)：刷题仓库，和本仓库无直接依赖。

## 说明

- 代码基于 PyTorch 2.x + Python 3.10+，CPU 可跑通全部示例。
- 图片存放在 `assets/<文章ID>/`，Markdown 里用相对路径引用，克隆后可离线查看。
- `html-snapshot/` 是 CSDN 原文备份；与 CSDN 不一致时，以本仓库 `notes/` 最新版本为准。
