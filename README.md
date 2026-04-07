# ZippedText

LLM 增强的无损中英文纯文本压缩工具。

基于自适应 PPM（Prediction by Partial Matching）算法与算术编码，专为中文、英文和数字混合文本设计。在纯离线模式下已超越 gzip 和 zstd 的压缩率；接入 DeepSeek API 后可进一步提升。

## 压缩效果

| 文本类型 | 原始 UTF-8 | gzip -9 | zstd -19 | **zippedtext** |
|---------|-----------|---------|----------|---------------|
| 中文 462 字 | 1,248 B | 781 B (0.63) | 785 B (0.63) | **679 B (0.54)** |
| 中英混合 2,622 字 | 4,925 B | 2,954 B (0.60) | 2,903 B (0.59) | **2,640 B (0.54)** |
| 重复文本 200 字 | 600 B | — | — | **32 B (0.05)** |

> 括号内为压缩比（越小越好）。

## 工作原理

1. **自适应 PPM 预测器**：维护多阶（order-0 到 order-4）字符级 n-gram 模型，在编码/解码过程中实时学习文本的统计规律。
2. **逃逸编码动态词汇**：无需预定义字符集。新字符首次出现时通过 `ESCAPE + Unicode codepoint` 编码，此后直接用自适应模型高效编码。
3. **32 位整数算术编码**：将预测概率转化为极致紧凑的比特流，数学上保证无损。
4. **LLM 增强**（可选）：接入 DeepSeek API 获取 logprobs，为算术编码提供更精准的概率分布，进一步压缩。

## 安装

### 从源码安装

```bash
git clone https://github.com/your-username/zippedtext.git
cd zippedtext
pip install .
```

### 开发模式安装

```bash
pip install -e ".[dev]"
```

### 依赖

- Python >= 3.12
- [click](https://click.palletsprojects.com/) — CLI 框架
- [openai](https://github.com/openai/openai-python) — DeepSeek API 客户端（OpenAI 兼容）
- [tqdm](https://github.com/tqdm/tqdm) — 进度条
- [zstandard](https://github.com/indygreg/python-zstandard) — zstd 压缩

## 快速开始

### 压缩文件

```bash
zippedtext c input.txt -o output.ztxt
```

### 解压文件

```bash
zippedtext d output.ztxt -o restored.txt
```

### 查看压缩文件信息

```bash
zippedtext info output.ztxt
```

输出示例：
```
File: output.ztxt
  Size:           679 bytes
  Mode:           offline
  Model:          deepseek-chat
  Token count:    462
  Original size:  1,248 bytes
  CRC32:          0x55971be1
  Model data:     0 bytes
  Compressed body:655 bytes
  Ratio:          0.544
```

### 基准测试

```bash
zippedtext bench input.txt
```

与 gzip、zstd 等算法对比压缩率：
```
Benchmarking: input.txt
  Text: 462 chars, 1,248 bytes (UTF-8)

  Method                        Size    Ratio
  ───────────────────────── ──────── ────────
  Raw UTF-8                    1,248    1.000
  gzip -9                        781    0.626
  zstd -19                       785    0.629
  zippedtext offline             679    0.544
```

## 配置 DeepSeek API（可选）

接入 DeepSeek API 可在压缩时利用 LLM 的语言预测能力，进一步提升压缩率。

### 1. 获取 API Key

访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并创建 API Key。

### 2. 设置环境变量

**Linux / macOS：**
```bash
export DEEPSEEK_API_KEY="sk-your-api-key-here"
```

**Windows (PowerShell)：**
```powershell
$env:DEEPSEEK_API_KEY = "sk-your-api-key-here"
```

**Windows (CMD)：**
```cmd
set DEEPSEEK_API_KEY=sk-your-api-key-here
```

也可以创建 `.env` 文件（不会被 git 追踪）：
```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### 3. 使用 LLM 增强压缩

```bash
# 通过环境变量
zippedtext c input.txt -o output.ztxt --mode online

# 通过命令行参数
zippedtext c input.txt --api-key sk-your-key --mode online

# 切换模型（默认 deepseek-chat）
zippedtext c input.txt --mode online --model deepseek-reasoner
```

> **注意**：不配置 API Key 也完全可以使用。离线模式（默认）不需要任何网络连接，已经优于 gzip/zstd。

## Python API

```python
from zippedtext.compressor import compress, decompress

# 压缩
text = "深度学习是人工智能的核心技术。"
data = compress(text)

# 解压
restored = decompress(data)
assert restored == text  # 无损保证
```

## 文件格式 (.ztxt)

```
Offset  Size    Field
──────  ────    ─────
0       4       Magic number (b'ZTXT')
4       1       Format version (0x01)
5       1       Mode (0x00=online, 0x01=offline)
6       2       Model ID
8       4       Token count
12      4       Original byte length (UTF-8)
16      4       CRC32 checksum
20      4       Model data length (0 for adaptive mode)
24      var     Compressed body (arithmetic coded bitstream)
```

## 项目结构

```
src/zippedtext/
├── __init__.py             # 版本号
├── __main__.py             # python -m zippedtext 入口
├── cli.py                  # CLI 命令 (compress/decompress/info/bench)
├── compressor.py           # 压缩/解压核心逻辑
├── arithmetic.py           # 32-bit 整数算术编码器/解码器
├── bitstream.py            # 比特流读写
├── format.py               # .ztxt 二进制文件格式
├── api_client.py           # DeepSeek API 客户端
├── tokenizer.py            # 字符级分词器
└── predictor/
    ├── base.py             # 预测器抽象基类
    ├── adaptive.py         # 自适应 PPM 预测器（核心）
    └── ngram.py            # N-gram 预测器
```

## 技术细节

### 算术编码

使用 Witten-Neal-Cleary 风格的 32 位整数算术编码。概率分布被量化为整数累积分布函数（CDF），保证：
- 每个符号频率 >= 1（防止零概率）
- CDF 总和精确等于 2^32
- 相同输入总是产生相同输出（确定性）

### 自适应预测

PPM 预测器维护 order-0（unigram）到 order-4 的上下文模型：
- 高阶上下文匹配时给予更高权重
- 通过观测次数自动调整各阶权重
- 编码器和解码器执行完全相同的更新步骤，保证同步

### 逃逸编码

字符首次出现时的编码策略：
- 编码 ESCAPE 符号（自适应概率）
- 编码字符所在范围（ASCII / CJK / 其他）— 3 选 1
- 编码范围内偏移量（均匀分布）
- CJK 字符首次编码约 17 bits，此后仅需约 8-12 bits

## 运行测试

```bash
# 算术编码单元测试
python tests/test_arithmetic.py

# 端到端无损测试
python tests/test_roundtrip.py

# 使用 pytest
pip install pytest
pytest tests/
```

## 路线图

- [ ] 接入 DeepSeek API logprobs 实现 online 模式的逐 token 算术编码
- [ ] Rust 核心加速（PyO3 绑定）
- [ ] 流式压缩/解压（支持大文件）
- [ ] 预置中文字符频率表（改善短文本压缩）
- [ ] 命令行自动补全

## License

[MIT](LICENSE)
