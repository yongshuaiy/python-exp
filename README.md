### transformer实验
实现的一个简易的Transformer模型，包含完整的训练、评估和消融实验框架。本项目旨在深入理解Transformer架构，并提供可配置的实验环境。

### 项目内容
包含完整的Encoder-Decoder结构、多头自注意力机制、位置编码、残差连接和层归一化、位置级FNN
完整的训练和评估流程、位置编码、多头注意力（head数量）、层数消融实验

### 项目结构
见arctecture.txt文件

### 实验环境要求
***所有超参数信息均在config.yaml文件中***
```
1. Python 3.13
2. PyTorch 2.0+、cuda13.0
3. NVIDIA GPU (8G以上，本实验全程在2070S训练)
4. 数据集为wikitext2
```

### 安装依赖
```
pip install -r requirements.txt
```
详细见requirements.txt文件

### 实验参数设置
```
详细见config.yaml文件
```

### 运行方式

1. 一键运行所有实验
```
python main.py
```

### 核心模块
```
1. MultiHeadAttention: 多头注意力机制，支持自注意力和交叉注意力
2. PositionWiseFFN: 位置级前馈网络
3. ResidualConnection: 残差连接和层归一化
4. PositionalEncoding: 正弦位置编码
5. EncoderLayer: Transformer编码器层
6. DecoderLayer: Transformer解码器层
```
### 消融实验
本实验包含多个消融实验——位置编码、多头注意力（head数）、层数
```
1. 位置编码消融：比较有无位置编码的性能差异
2. 注意力头数消融：测试不同注意力头数1, 2, 4, 8的影响
3. 层数消融：测试不同层数1, 2, 4, 6的影响
```



