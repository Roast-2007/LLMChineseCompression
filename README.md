# LLMChineseCompression

LLM 增强的无损中英文纯文本压缩工具。

基于自适应 PPM（Prediction by Partial Matching）算法与算术编码，专为中文、英文和数字混合文本设计。在纯离线模式下已超越 gzip 和 zstd 的压缩率；接入 LLM API 后可进一步提升。

**v0.3.2 新增**：引入 **structured online** 主路径、`.ztxt` v3 section 化格式、segment router、LLM 辅助术语/短语字典、API-free structured 解压、legacy online 冻结与兼容保留。

**v0.3.1 新增**：在线模式性能优化（压缩速度 13× 提升、解压缩速度 1200× 提升）、预测缓存嵌入（解压无需 API）、智能回退（自动选择最优模式）。

**v0.3.0 新增**：中文字符频率先验、短语级编码、代码生成模式（实验性）、多平台 API 支持（SiliconFlow 硅基流动等）、交互式配置系统。

## 当前 online 模式说明

项目现在同时保留两类 online 路径：

1. **structured online（默认）**
   - 压缩期调用 LLM 做结构分析；
   - 让 LLM 参与短语/术语提示、语言片段提示、segment routing；
   - 结果写入 `.ztxt` v3 的结构化 side info section；
   - **解压不依赖 API**。

2. **legacy online（兼容模式）**
   - `--sub-mode char`
   - `--sub-mode token`
   - 保留旧版 next-token / char boost + prediction cache 路径，用于对比和兼容旧思路。

一句话：

> v0.3.2 开始，online mode 不再只是“让 LLM 续写然后给 PPM 打补丁”，而是让 LLM 开始参与分段、字典构建、结构化 side info 设计与压缩模式选择。

## 工作原理

1. **自适应 PPM 预测器**：维护多阶（order-0 到 order-6 可配置）字符级 n-gram 模型，在编码/解码过程中实时学习文本的统计规律。
2. **中文字符频率先验**：预置 top-3000 中文字符频率表作为 warm start，显著改善短文本压缩。
3. **短语级编码**：自动识别高频短语（2-8 字符），将其作为单一符号编码。
4. **structured online（当前主路径）**：
   - LLM 一次性分析全文；
   - 生成字符频率、短语/术语候选、语言片段提示、template hints；
   - 本地分段，并用 gain estimator 在 literal / phrase / template 之间做净收益选路；
   - template route 采用 `template_id + slot values + residual`，residual 继续复用现有 literal / phrase coder；
   - analysis / dictionary / templates / segment metadata / stats 写入 `.ztxt` v3 typed sections；
   - 解压时不访问远端 API，只依赖文件内 side info 和确定性本地 coder。
5. **legacy online char/token**：保留旧版 prediction-cache 方案，用于兼容和效果对比。
6. **代码生成模式**（实验性）：LLM 识别可用 Python 表达式表示的文本片段，将这些片段存储为代码而非压缩数据。

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
zippedtext config init
```

### 压缩文件

```bash
# 离线模式（默认，无需网络）
zippedtext c input.txt -o output.ztxt

# structured online（默认在线子模式）
zippedtext c input.txt -o output.ztxt --mode online --api-key sk-your-key

# 显式指定 structured
zippedtext c input.txt -o output.ztxt --mode online --sub-mode structured --api-key sk-your-key

# legacy online char
zippedtext c input.txt -o output.ztxt --mode online --sub-mode char --api-key sk-your-key

# legacy online token
zippedtext c input.txt -o output.ztxt --mode online --sub-mode token --api-key sk-your-key

# 代码生成模式（实验性，需要 API）
zippedtext c input.txt -o output.ztxt --mode codegen --api-key sk-your-key
```

### 解压文件

```bash
# 离线模式文件
zippedtext d output.ztxt -o restored.txt

# structured online 文件 / v0.3.1+ 带 cache 的 legacy online 文件
zippedtext d output.ztxt -o restored.txt

# 只有旧版未嵌入 cache 的 legacy online 文件才可能需要 API
zippedtext d output.ztxt -o restored.txt --api-key sk-your-key
```

### 查看压缩文件信息

```bash
zippedtext info output.ztxt
```

### 基准测试

```bash
zippedtext bench input.txt
zippedtext bench input.txt --api-key sk-your-key
```

## CLI 命令参考

### `zippedtext c` — 压缩

```text
参数:
  INPUT_FILE                输入文本文件

选项:
  -o, --output PATH         输出 .ztxt 文件路径
  --mode [offline|online|codegen]  压缩模式（默认: offline）
  --sub-mode [structured|char|token]  在线子模式（默认: structured）
  --model TEXT              模型名称（覆盖配置）
  --api-key TEXT            API Key（覆盖配置/环境变量）
  --base-url TEXT           API 地址（覆盖配置）
  --max-order [4|5|6]       PPM 上下文深度（默认: 4）
  --no-priors               禁用中文频率先验
  --phrases                 启用短语级编码（实验性，仅 offline）
```

### `zippedtext d` — 解压

```text
参数:
  INPUT_FILE                输入 .ztxt 文件

选项:
  -o, --output PATH         输出文本文件路径
  --api-key TEXT            API Key（仅旧版 legacy online 文件可能需要）
  --base-url TEXT           API 地址
```

### `zippedtext info` — 文件信息

会显示：
- 格式版本（v2 / v3）
- online 路径类型（structured / legacy-char / legacy-token）
- structured online 的 analysis / dictionary / templates / segment / stats 拆解
- route 分布、template hit、residual bytes、side info total
- 是否可 API-free 解压

### `zippedtext bench` — 基准测试

会对比：
- gzip -9
- zstd -19
- zippedtext offline
- zippedtext online (structured)
- zippedtext online (legacy char)
- zippedtext online (legacy token)

其中 structured online 会额外显示：
- side info / payload / residual 成本
- route 分布
- 若 whole-file 最终回退 offline，则显示 fallback 差值与 loss reason

## Python API

```python
from zippedtext.compressor import compress, decompress
from zippedtext.api_client import ApiClient

text = "深度学习是人工智能的核心技术。"

# offline
data = compress(text)
assert decompress(data) == text

client = ApiClient(
    api_key="sk-your-key",
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
)

# structured online
data = compress(text, mode="online", api_client=client, sub_mode="structured")
assert decompress(data) == text

# legacy online char
data = compress(text, mode="online", api_client=client, sub_mode="char")
assert decompress(data) == text
```

## 文件格式 (.ztxt)

### v2 格式（legacy offline / legacy online / codegen）

```text
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

### v3 格式（structured online）

`.ztxt` v3 采用 section 化设计，当前至少包含：
- analysis section
- phrase / dictionary section
- template catalog section
- segment metadata section
- route stats section
- payload body

section entry 自带 flags，可选择 raw / zstd codec；读取时会保留 section flags 与 stored size。

这让 online mode 可以存储**压缩友好的结构化决策信息**，而不是直接缓存原始生成文本。

## 项目结构

```text
src/zippedtext/
├── __init__.py             # 版本号
├── __main__.py             # python -m zippedtext 入口
├── cli.py                  # CLI 命令定义
├── cli_config.py           # config 子命令组
├── config.py               # 配置系统
├── provider.py             # API 平台预设
├── compressor.py           # 压缩/解压编排器
├── encoder.py              # 编码逻辑
├── decoder.py              # 解码逻辑
├── cdf_utils.py            # CDF 工具
├── codegen.py              # 代码生成模式
├── arithmetic.py           # 32-bit 整数算术编码器/解码器
├── bitstream.py            # 比特流读写
├── format.py               # .ztxt 二进制格式 (v1 + v2 + v3)
├── api_client.py           # API 客户端（OpenAI 协议兼容）
├── online_manifest.py      # structured online manifest / stats
├── sideinfo_codec.py       # section codec / side info helpers
├── template_codec.py       # template catalog / template payload codec
├── residual.py             # residual payload packing / reuse of literal+phrase coder
├── gain_estimator.py       # segment gain estimation / route scoring
├── segment.py              # segment splitter / classifier
├── router.py               # segment router / gain-based selection
├── term_dictionary.py      # LLM + heuristic phrase dictionary builder
└── predictor/
    ├── adaptive.py         # 自适应 PPM 预测器（核心）
    ├── priors.py           # 中文字符频率先验
    ├── phrases.py          # 短语级编码
    └── llm.py              # legacy online 预测器
```

## 运行测试

```bash
# 全部测试
pytest tests/

# structured online 新增测试
pytest tests/test_analysis_manifest.py tests/test_router.py tests/test_online_structured.py tests/test_format_v3.py

# legacy online mock 测试
pytest tests/test_online.py

# 在线模式集成测试（需要 API Key）
DEEPSEEK_API_KEY=sk-your-key pytest tests/test_online_integration.py -v -s
```

## 路线图

- [x] ~~接入 DeepSeek API logprobs 实现 online 模式~~ (v0.2.0)
- [x] ~~预置中文字符频率表~~ (v0.3.0)
- [x] ~~短语级编码~~ (v0.3.0)
- [x] ~~多平台 API 支持~~ (v0.3.0)
- [x] ~~代码生成模式~~ (v0.3.0)
- [x] ~~交互式配置系统~~ (v0.3.0)
- [x] ~~在线模式性能优化~~ (v0.3.1)
- [x] ~~prediction cache 嵌入~~ (v0.3.1)
- [x] ~~冻结 legacy online char/token~~ (v0.3.2)
- [x] ~~引入 structured online 主路径~~ (v0.3.2)
- [x] ~~引入 `.ztxt` v3 section 化格式~~ (v0.3.2)
- [x] ~~segment router~~ (v0.3.2)
- [x] ~~LLM 辅助 term/phrase dictionary~~ (v0.3.2)
- [ ] template codec
- [ ] residual architecture
- [ ] mixture-of-experts probability layer
- [ ] 本地确定性模型
- [ ] Rust 核心加速（PyO3）

## 更新日志

### v0.3.2 — structured online 初步落地

- 新增 **structured online** 主路径，并设为默认 online 子模式
- 新增 `.ztxt` **v3** section 化格式
- 新增 `segment.py` 与 `router.py`，支持 segment router
- 新增 `online_manifest.py`，存储 analysis / segment / route stats
- 新增 `term_dictionary.py`，把 LLM 分析结果接入短语/术语字典构建
- `ApiClient.analyze_text()` 改为返回结构化 manifest，而不是裸 JSON
- structured online 解压 **不依赖远端 API**
- legacy `char` / `token` online 冻结为兼容路径
- `bench` 现在可对比 structured / legacy char / legacy token
- `info` 现在可显示 structured online 的 v3 section 信息

### v0.3.1 — 在线模式性能优化

- `CHUNK_CHARS` 从 20 提升至 200（API 调用减少 10 倍）
- prediction cache 嵌入 `.ztxt` 文件，legacy online 解压无需 API
- 在线模式结果若大于离线模式则自动回退 offline
- 扩展 model_data 存储 chunk_chars、max_tokens、prediction_cache
- 修复 DeepSeek API 长上下文非确定性问题

### v0.3.0 — 压缩率优化 + 多平台支持 + 代码生成模式

- 中文字符频率先验
- 短语级编码
- order-5 / order-6 支持
- 代码生成模式（实验性）
- 多平台 API 支持
- 交互式配置系统
- `.ztxt` v2

### v0.2.0 — DeepSeek API 在线模式

- 新增 `--mode online`
- 支持 `char` / `token` 子模式
- 新增在线模式 mock 测试和集成测试

### v0.1.0 — 初始版本

- 自适应 PPM + 32 位算术编码
- 逃逸编码动态词汇
- `.ztxt` 二进制格式 v1
- CLI: compress / decompress / info / bench

## License

[MIT](LICENSE)
