# LLMChineseCompression

LLM 增强的无损中英文纯文本压缩工具。

基于自适应 PPM（Prediction by Partial Matching）算法与算术编码，专为中文、英文和数字混合文本设计。在纯离线模式下已超越 gzip 和 zstd 的压缩率；接入 DeepSeek API 后可进一步提升。

## 压缩效果

| 文本类型 | 原始 UTF-8 | gzip -9 | zstd -19 | **offline** | **online (char)** |
|---------|-----------|---------|----------|-------------|-------------------|
| 中文 462 字 | 1,248 B | 781 B (0.63) | 785 B (0.63) | **679 B (0.54)** | **661 B (0.53)** |
| 中英混合 2,622 字 | 4,925 B | 2,954 B (0.60) | 2,903 B (0.59) | **2,640 B (0.54)** | — |
| 重复文本 200 字 | 600 B | — | — | **32 B (0.05)** | — |

> 括号内为压缩比（越小越好）。online 模式需要 DeepSeek API Key。

## 工作原理

1. **自适应 PPM 预测器**：维护多阶（order-0 到 order-4）字符级 n-gram 模型，在编码/解码过程中实时学习文本的统计规律。
2. **逃逸编码动态词汇**：无需预定义字符集。新字符首次出现时通过 `ESCAPE + Unicode codepoint` 编码，此后直接用自适应模型高效编码。
3. **32 位整数算术编码**：将预测概率转化为极致紧凑的比特流，数学上保证无损。
4. **LLM 在线增强**（v0.2.0 新增）：接入 DeepSeek API 生成文本续写预测，利用 LLM 的语言知识增强字符概率分布，进一步压缩。支持两种子模式：
   - **char**（默认）：字符级预测增强，LLM 预测准确时大幅降低编码位数
   - **token**（实验性）：token 级匹配编码，匹配的 token 几乎零成本编码

## 安装

### 从源码安装

```bash
git clone https://github.com/Roast-2007/LLMChineseCompression.git
cd LLMChineseCompression
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

### 压缩文件（离线模式）

```bash
zippedtext c input.txt -o output.ztxt
```

### 压缩文件（在线模式 — 需要 API Key）

```bash
# 字符级在线模式（默认子模式）
zippedtext c input.txt -o output.ztxt --mode online --api-key sk-your-key

# token 级在线模式（实验性）
zippedtext c input.txt -o output.ztxt --mode online --sub-mode token --api-key sk-your-key
```

### 解压文件

```bash
# 离线模式文件 — 无需 API Key
zippedtext d output.ztxt -o restored.txt

# 在线模式文件 — 需要 API Key（用于重现相同的 LLM 预测）
zippedtext d output.ztxt -o restored.txt --api-key sk-your-key
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
# 仅离线模式
zippedtext bench input.txt

# 包含在线模式（需要 API Key）
zippedtext bench input.txt --api-key sk-your-key
```

## 配置 DeepSeek API

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

# 使用 token 级子模式（实验性）
zippedtext c input.txt --mode online --sub-mode token

# 切换模型（默认 deepseek-chat）
zippedtext c input.txt --mode online --model deepseek-reasoner
```

> **注意**：不配置 API Key 也完全可以使用。离线模式（默认）不需要任何网络连接，已经优于 gzip/zstd。

### 在线模式说明

- **压缩和解压都需要 API 访问**：在线模式的解压需要调用相同的 API 重现 LLM 预测
- **API 确定性**：使用 `temperature=0` + `seed=42` 保证编码/解码的预测完全一致
- **API 费用**：每次压缩/解压约消耗数十次 API 调用，请关注费用
- **当前限制**：受 Chat API `logprobs` 精度限制，在线模式相比离线约提升 2-5%。未来版本通过本地模型推理可大幅提升

## Python API

```python
from zippedtext.compressor import compress, decompress

# 离线压缩
text = "深度学习是人工智能的核心技术。"
data = compress(text)

# 在线压缩（需要 API Key）
from zippedtext.api_client import DeepSeekClient
client = DeepSeekClient(api_key="sk-your-key")
data = compress(text, mode="online", api_client=client, sub_mode="char")

# 解压
restored = decompress(data)  # 离线文件
restored = decompress(data, api_client=client)  # 在线文件
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
20      4       Model data length (0 for offline, >0 for online)
24      var     Model data (online: sub-mode byte + model name)
24+N    var     Compressed body (arithmetic coded bitstream)
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
    ├── llm.py              # LLM 在线预测器 (v0.2.0)
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

### LLM 在线增强 (v0.2.0)

在线模式通过 DeepSeek API 获取文本续写预测，利用 LLM 的语言知识增强编码：

- **字符级 (char)**：每 20 个字符调用一次 API 获取续写预测，对预测正确的字符大幅提升概率（20x boost），自动在预测偏离后停止增强（stop-on-mismatch），避免错误预测造成编码膨胀
- **token 级 (token)**：匹配 API 生成的 token 与实际文本，匹配的 token 通过 logprobs CDF 高效编码，不匹配时回退到字符级 PPM

### 逃逸编码

字符首次出现时的编码策略：
- 编码 ESCAPE 符号（自适应概率）
- 编码字符所在范围（ASCII / CJK / 其他）— 3 选 1
- 编码范围内偏移量（均匀分布）
- CJK 字符首次编码约 17 bits，此后仅需约 8-12 bits

## 运行测试

```bash
# 全部测试
pytest tests/

# 仅离线模式测试
pytest tests/test_arithmetic.py tests/test_roundtrip.py

# 在线模式 mock 测试
pytest tests/test_online.py

# 在线模式集成测试（需要 API Key）
DEEPSEEK_API_KEY=sk-your-key pytest tests/test_online_integration.py -v -s
```

## 路线图

- [x] ~~接入 DeepSeek API logprobs 实现 online 模式~~ (v0.2.0)
- [ ] 预置中文字符频率表（改善短文本压缩）
- [ ] 短语级编码（高频短语作为单一符号）
- [ ] Rust 核心加速（PyO3 绑定）
- [ ] 本地模型推理（大幅提升在线模式压缩率）
- [ ] 流式压缩/解压（支持大文件）
- [ ] 命令行自动补全

## 更新日志

### v0.2.0 — DeepSeek API 在线模式

- 新增 `--mode online` 在线压缩模式，接入 DeepSeek Chat API
- 支持两种子模式：`--sub-mode char`（默认）和 `--sub-mode token`（实验性）
- 字符级模式：LLM 预测增强 + stop-on-mismatch 策略
- token 级模式：API token 匹配编码 + 字符级回退
- API 客户端：自动重试、模型版本追踪、确定性生成
- 新增在线模式 mock 测试和集成测试
- bench 命令支持在线模式对比

### v0.1.0 — 初始版本

- 自适应 PPM + 32 位算术编码
- 逃逸编码动态词汇
- .ztxt 二进制格式 v1
- CLI: compress / decompress / info / bench
- 离线模式中文压缩率 0.54，优于 gzip/zstd

## License

[MIT](LICENSE)
