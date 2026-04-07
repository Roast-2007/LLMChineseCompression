# LLMChineseCompression

LLM 增强的无损中英文纯文本压缩工具。

基于自适应 PPM（Prediction by Partial Matching）算法与算术编码，专为中文、英文和数字混合文本设计。在纯离线模式下已超越 gzip 和 zstd 的压缩率；接入 LLM API 后可进一步提升。

**v0.3.0 新增**：中文字符频率先验、短语级编码、代码生成模式（实验性）、多平台 API 支持（SiliconFlow 硅基流动等）、交互式配置系统。

## 压缩效果

| 文本类型 | 原始 UTF-8 | gzip -9 | zstd -19 | **offline** | **online (char)** |
|---------|-----------|---------|----------|-------------|-------------------|
| 中文 462 字 | 1,248 B | 781 B (0.63) | 785 B (0.63) | **679 B (0.54)** | **661 B (0.53)** |
| 中英混合 2,622 字 | 4,925 B | 2,954 B (0.60) | 2,903 B (0.59) | **2,640 B (0.54)** | — |
| 重复文本 200 字 | 600 B | — | — | **32 B (0.05)** | — |

> 括号内为压缩比（越小越好）。v0.3.0 的短语级编码和频率先验可进一步改善离线模式压缩率。

## 工作原理

1. **自适应 PPM 预测器**：维护多阶（order-0 到 order-6 可配置）字符级 n-gram 模型，在编码/解码过程中实时学习文本的统计规律。
2. **中文字符频率先验**（v0.3.0）：预置 top-3000 中文字符频率表作为 warm start，显著改善短文本压缩。
3. **短语级编码**（v0.3.0）：自动识别高频短语（2-8 字符），将其作为单一符号编码。
4. **逃逸编码动态词汇**：无需预定义字符集。新字符首次出现时通过 `ESCAPE + Unicode codepoint` 编码，此后直接用自适应模型高效编码。
5. **32 位整数算术编码**：将预测概率转化为极致紧凑的比特流，数学上保证无损。
6. **LLM 在线增强**（v0.2.0）：接入 LLM API 生成文本续写预测，利用 LLM 的语言知识增强字符概率分布。支持两种子模式：
   - **char**（默认）：字符级预测增强，LLM 预测准确时大幅降低编码位数
   - **token**（实验性）：token 级匹配编码，匹配的 token 几乎零成本编码
7. **代码生成模式**（v0.3.0，实验性）：LLM 识别可用 Python 表达式表示的文本片段，将这些片段存储为代码而非压缩数据。

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
- [openai](https://github.com/openai/openai-python) — API 客户端（OpenAI 协议兼容）
- [tqdm](https://github.com/tqdm/tqdm) — 进度条
- [zstandard](https://github.com/indygreg/python-zstandard) — zstd 压缩

## 快速开始

### 配置 API（可选）

```bash
# 交互式配置向导 — 引导选择平台、输入密钥、选择模型
zippedtext config init
```

### 压缩文件

```bash
# 离线模式（默认，无需网络）
zippedtext c input.txt -o output.ztxt

# 在线模式 — 使用已配置的 API
zippedtext c input.txt -o output.ztxt --mode online

# 在线模式 — 直接指定 API Key
zippedtext c input.txt -o output.ztxt --mode online --api-key sk-your-key

# 代码生成模式（实验性，需要 API）
zippedtext c input.txt -o output.ztxt --mode codegen --api-key sk-your-key
```

### 解压文件

```bash
# 离线模式文件
zippedtext d output.ztxt -o restored.txt

# 在线/codegen 模式文件（需要 API 访问）
zippedtext d output.ztxt -o restored.txt --api-key sk-your-key
```

### 查看压缩文件信息

```bash
zippedtext info output.ztxt
```

### 基准测试

```bash
zippedtext bench input.txt
zippedtext bench input.txt --api-key sk-your-key  # 含在线模式
```

## CLI 命令参考

### `zippedtext c` — 压缩

```
参数:
  INPUT_FILE                输入文本文件

选项:
  -o, --output PATH         输出 .ztxt 文件路径
  --mode [offline|online|codegen]  压缩模式（默认: offline）
  --sub-mode [char|token]   在线子模式（默认: char）
  --model TEXT              模型名称（覆盖配置）
  --api-key TEXT            API Key（覆盖配置/环境变量）
  --base-url TEXT           API 地址（覆盖配置）
  --max-order [4|5|6]       PPM 上下文深度（默认: 4）
  --no-priors               禁用中文频率先验
  --no-phrases              禁用短语级编码
```

### `zippedtext d` — 解压

```
参数:
  INPUT_FILE                输入 .ztxt 文件

选项:
  -o, --output PATH         输出文本文件路径
  --api-key TEXT            API Key（在线模式文件需要）
  --base-url TEXT           API 地址
```

### `zippedtext info` — 文件信息

```
参数:
  INPUT_FILE                .ztxt 文件路径
```

### `zippedtext bench` — 基准测试

```
参数:
  INPUT_FILE                输入文本文件

选项:
  --api-key TEXT            API Key（可选，启用在线模式对比）
  --base-url TEXT           API 地址
  --model TEXT              模型名称
```

### `zippedtext config` — 配置管理

```bash
# 交互式配置向导
zippedtext config init

# 查看当前配置
zippedtext config show

# 设置单个配置项
zippedtext config set model deepseek-chat
zippedtext config set base_url https://api.siliconflow.cn/v1
zippedtext config set api_key sk-your-key

# 获取可用模型列表
zippedtext config models
```

配置文件位置：`~/.zippedtext/config.json`

配置优先级：CLI 参数 > 环境变量 > 配置文件 > 默认值

支持的环境变量：`ZIPPEDTEXT_API_KEY`、`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`、`OPENAI_API_KEY`

## 多平台 API 支持

v0.3.0 支持任何 OpenAI 协议兼容的 API 平台：

| 平台 | Base URL | 说明 |
|------|----------|------|
| DeepSeek | `https://api.deepseek.com` | 默认平台 |
| SiliconFlow 硅基流动 | `https://api.siliconflow.cn/v1` | 国内平台，模型丰富 |
| OpenAI | `https://api.openai.com/v1` | GPT 系列模型 |
| 自定义 | 任意 URL | 任何 OpenAI 协议兼容 API |

### 使用 SiliconFlow 示例

```bash
# 方法 1：交互式配置
zippedtext config init
# 选择 "SiliconFlow 硅基流动" → 输入 API Key → 选择模型

# 方法 2：手动配置
zippedtext config set base_url https://api.siliconflow.cn/v1
zippedtext config set api_key your-siliconflow-key
zippedtext config models  # 查看可用模型并选择

# 方法 3：命令行直接指定
zippedtext c input.txt --mode online \
  --base-url https://api.siliconflow.cn/v1 \
  --api-key your-key \
  --model Qwen/Qwen2.5-7B-Instruct
```

## Python API

```python
from zippedtext.compressor import compress, decompress

# 离线压缩（默认启用频率先验 + 短语编码）
text = "深度学习是人工智能的核心技术。"
data = compress(text)
restored = decompress(data)
assert restored == text

# 自定义选项
data = compress(text, use_priors=False, use_phrases=False, max_order=5)

# 在线压缩（任何 OpenAI 协议兼容 API）
from zippedtext.api_client import ApiClient
client = ApiClient(
    api_key="sk-your-key",
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
)
data = compress(text, mode="online", api_client=client, sub_mode="char")
restored = decompress(data, api_client=client)

# 代码生成模式（实验性）
data = compress(text, mode="codegen", api_client=client)
restored = decompress(data)  # codegen 解压不需要 API

# 配置系统
from zippedtext.config import resolve_config, save_config, AppConfig
cfg = resolve_config(cli_api_key="sk-...", cli_model="deepseek-chat")
```

## 文件格式 (.ztxt)

### v2 格式（v0.3.0 起，32 字节 header）

```
Offset  Size    Field
──────  ────    ─────
0       4       Magic number (b'ZTXT')
4       1       Format version (0x02)
5       1       Mode (0x00=online, 0x01=offline, 0x02=codegen)
6       1       Flags (bit0: phrases, bit1: priors)
7       1       Max PPM order (4/5/6)
8       4       Token count
12      4       Original byte length (UTF-8)
16      4       CRC32 checksum
20      4       Model data length
24      4       Phrase table length
28      4       Reserved
32+     var     Model data
32+M    var     Phrase table
32+M+P  var     Compressed body
```

> v1 格式（v0.1.0-v0.2.0）仍可正常解压，向后兼容。

## 项目结构

```
src/zippedtext/
├── __init__.py             # 版本号
├── __main__.py             # python -m zippedtext 入口
├── cli.py                  # CLI 命令定义
├── cli_config.py           # config 子命令组 (v0.3.0)
├── config.py               # 配置系统 (v0.3.0)
├── provider.py             # API 平台预设 (v0.3.0)
├── compressor.py           # 压缩/解压编排器
├── encoder.py              # 编码逻辑 (v0.3.0 拆分)
├── decoder.py              # 解码逻辑 (v0.3.0 拆分)
├── cdf_utils.py            # CDF 工具 (v0.3.0)
├── codegen.py              # 代码生成模式 (v0.3.0)
├── arithmetic.py           # 32-bit 整数算术编码器/解码器
├── bitstream.py            # 比特流读写
├── format.py               # .ztxt 二进制格式 (v1 + v2)
├── api_client.py           # API 客户端（OpenAI 协议兼容）
├── tokenizer.py            # 字符级分词器
└── predictor/
    ├── base.py             # 预测器抽象基类
    ├── adaptive.py         # 自适应 PPM 预测器（核心）
    ├── priors.py           # 中文字符频率先验 (v0.3.0)
    ├── phrases.py          # 短语级编码 (v0.3.0)
    ├── llm.py              # LLM 在线预测器
    └── ngram.py            # N-gram 预测器
```

## 运行测试

```bash
# 全部测试
pytest tests/

# 仅离线模式测试
pytest tests/test_arithmetic.py tests/test_roundtrip.py

# 在线模式 mock 测试
pytest tests/test_online.py

# v0.3.0 新增测试
pytest tests/test_format_v2.py tests/test_config.py tests/test_phrases.py tests/test_codegen.py

# 在线模式集成测试（需要 API Key）
DEEPSEEK_API_KEY=sk-your-key pytest tests/test_online_integration.py -v -s
```

## 路线图

- [x] ~~接入 DeepSeek API logprobs 实现 online 模式~~ (v0.2.0)
- [x] ~~预置中文字符频率表~~ (v0.3.0)
- [x] ~~短语级编码~~ (v0.3.0)
- [x] ~~多平台 API 支持（SiliconFlow 等）~~ (v0.3.0)
- [x] ~~代码生成模式（实验性）~~ (v0.3.0)
- [x] ~~交互式配置系统~~ (v0.3.0)
- [x] ~~order-5/order-6 支持~~ (v0.3.0)
- [ ] Rust 核心加速（PyO3 绑定）
- [ ] 本地模型推理（大幅提升在线模式压缩率）
- [ ] 流式压缩/解压（支持大文件）
- [ ] 命令行自动补全
- [ ] PyPI 发布

## 更新日志

### v0.3.0 — 压缩率优化 + 多平台支持 + 代码生成模式

- **压缩率优化**：
  - 预置 top-3000 中文字符频率表作为 PPM 先验，改善短文本压缩
  - 短语级编码：自动识别高频短语（2-8 字符）作为单一符号
  - 支持 order-5/order-6 PPM 上下文（`--max-order 5`）
- **代码生成模式**（实验性）：`--mode codegen`，LLM 识别可用代码表示的文本片段
- **多平台 API 支持**：支持 DeepSeek、SiliconFlow 硅基流动、OpenAI 及任何 OpenAI 协议兼容 API
- **交互式配置系统**：`zippedtext config init/show/set/models`，本地持久化配置
- **.ztxt 格式 v2**：32 字节 header，支持 flags、phrase table、任意模型名；向后兼容 v1
- **代码重构**：compressor.py 拆分为 encoder.py + decoder.py + cdf_utils.py；DeepSeekClient 重命名为 ApiClient
- 新增测试：test_format_v2.py、test_config.py、test_phrases.py、test_codegen.py

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
