# LLMChineseCompression

LLM 增强的无损中英文纯文本压缩工具。

基于自适应 PPM（Prediction by Partial Matching）算法与算术编码，专为中文、英文和数字混合文本设计。在纯离线模式下已超越 gzip 和 zstd 的压缩率；预期在v0.5.0前后接入 LLM API 可进一步提升压缩率，当前仍在探索合适的llm深度嵌入方式。

**v0.3.6 新增**：Record Family 扩展——新增 `RecordGroup` 数据结构实现连续结构化段落分组、`record` 模板 kind 支持多行记录模板（JSON-encoded per-line skeleton）、typed residual 新增 `RESIDUAL_TYPED` 路由支持 enum/version/path_or_url/number_with_unit/identifier 类型化残差编码、router 接受 `record_groups` 参数为族级收益优化奠基、新增 benchmark matrix 含 6 类结构化样本（API 文档/配置/更新日志/FAQ/双语术语/混合散文）。

**v0.3.5 新增**：代码质量与健壮性加固——`_merge_priors` 浮点精度过滤修复、`_match_key_value` 嵌套括号安全处理、structured 路径前轻量启发式预检（小文本跳过 API 调用）、`probs_to_cdf` redistribution 极端情况兜底、`_looks_like_config` 行级模式检测修复、`router.py` 消除 `detect_template` 重复调用、`_count_all_phrase_occurrences` 单次全文扫描优化、内部函数重命名为公共 API（`get_priors` / `structured_compress`）。

**v0.3.4 新增**：structured online 第三阶段聚焦 schema-seeded typed slot codec：analysis manifest 新增 `document_family` / `block_families` / `field_schemas` / `slot_hints` / `enum_candidates`，template slot 开始支持 `identifier` / `version` / `path_or_url` / `enum` / `number_with_unit`，`info` / `bench` 新增 typed slot 与 template family 可观测性，`bench` 现在同时展示 public structured 结果和 raw structured diagnostic。

**v0.3.3 新增**：structured online 第二阶段优化：analysis priors 真正接入 structured coder、list/table/config 行级切分、hint-aware template detection、phrase 排序增强、更细 fallback reason 与 structured API smoke test。

> v0.3.5 在 structured online 主路径功能完整的基础上，聚焦代码健壮性和性能优化：修复了 priors 合并、模板解析、CDF 归一化等边界条件问题，消除了重复 API 调用和 O(n*m*k) 扫描开销，同时将小文本/低结构化文本的 API 调用智能跳过，避免不必要的费用支出。

## 当前 online 模式说明

项目现在同时保留两类 online 路径：

1. **structured online（默认）**
   - 压缩期调用 LLM 做结构分析；
   - analysis manifest 不再只有 `template_hints`，而是可携带 document/block/schema/slot hints；
   - analysis 中的字符先验会与本地 priors 合并后参与 structured literal / phrase / template residual coder；
   - `list` / `table` / `config` 块会优先按行切分，提高 template route 命中机会；
   - template route 仍然采用 `template_id + slot payload + residual`，但 slot 已可优先走 typed slot codec；
   - 结果写入 `.ztxt` v3 的结构化 side info section；
   - **解压不依赖 API**。

2. **legacy online（兼容模式）**
   - `--sub-mode char`
   - `--sub-mode token`
   - 保留旧版 next-token / char boost + prediction cache 路径，用于对比和兼容旧思路。

## 工作原理

1. **自适应 PPM 预测器**：维护多阶（order-0 到 order-6 可配置）字符级 n-gram 模型，在编码/解码过程中实时学习文本的统计规律。
2. **中文字符频率先验**：预置 top-3000 中文字符频率表作为 warm start，显著改善短文本压缩。
3. **短语级编码**：自动识别高频短语（2-8 字符），将其作为单一符号编码。
4. **structured online（当前主路径）**：
   - LLM 一次性分析全文；
   - 生成字符频率、短语/术语候选、语言片段提示、template hints，以及 document/schema/slot 级提示；
   - 本地分段；对 `list` / `table` / `config` 优先做行级切分；
   - analysis 中的字符先验会与本地 priors 融合，进入 structured literal / phrase / template residual 编解码；
   - phrase table 会综合 heuristic phrases、LLM phrase hints、top bigrams 与 char frequencies 做排序；
   - template route 仍然采用 `template_id + typed slot payload + residual`，slot value 可按 `version` / `path_or_url` / `enum` / `number_with_unit` / `identifier` 等类型压缩；
   - 本地用 gain estimator 在 literal / phrase / template 之间做净收益选路，并记录更细的 fallback reason；
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

其中 `bench` 现在会同时显示：
- public `compress(..., mode="online", sub_mode="structured")` 的最终结果
- raw structured diagnostic（便于观察 whole-file fallback 之前的真实 structured payload / side info 结构）

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
- route 分布、template hit、typed slot / typed template 统计、template family 统计、residual bytes、side info total
- 是否可 API-free 解压

### `zippedtext bench` — 基准测试

会对比：
- gzip -9
- zstd -19
- zippedtext offline
- zippedtext online (structured)
- structured raw diagnostic
- zippedtext online (legacy char)
- zippedtext online (legacy token)

其中 structured online 会额外显示：
- side info / payload / residual 成本
- typed slot / typed template 统计
- template family 分布
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

analysis section 现在可增量承载：
- `document_family`
- `block_families`
- `field_schemas`
- `slot_hints`
- `enum_candidates`

这让 online mode 存储的是**压缩友好的结构化决策信息**，而不是原始生成文本本身。

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
├── template_codec.py       # template catalog / typed slot payload codec
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

# structured online 重点回归
pytest tests/test_analysis_manifest.py tests/test_router.py tests/test_template_codec.py tests/test_residual.py tests/test_gain_estimator.py tests/test_online_structured.py tests/test_format_v3.py tests/test_cli.py

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
- [x] ~~template codec~~ (v0.3.3)
- [x] ~~residual architecture~~ (v0.3.3)
- [x] ~~schema-seeded typed slot codec~~ (v0.3.4)
- [x] ~~bench / info / CLI 可观测性补强~~ (v0.3.4)
- [x] ~~code quality + robustness hardening~~ (v0.3.5)
- [x] ~~record family expansion~~ (v0.3.6)
- [ ] document-level family clustering + global gain optimization 深化
- [ ] 本地确定性模型
- [ ] Rust 核心加速（PyO3）

### v0.3.6 — record family expansion

- 新增 `RecordGroup` 数据结构（`segment.py`）：`kind` / `segment_indices` / `family` / `text_span`
- 新增 `group_record_groups()` 函数：连续同类型结构化段落聚类
- `template_codec.py` 新增 `record` 模板 kind（ID: 3）
- `TemplateMatch` 新增 `skeleton_lines` 字段支持多行骨架
- `TemplateCatalog` 序列化/反序列化支持 record 模板的 JSON-encoded per-line skeleton
- 新增 `_render_record_template()` 多行渲染逻辑
- 新增 `_match_record_template()` 多行记录模板检测
- 新增 `_scan_record_templates()` catalog 扫描（Phase 4 启用）
- `residual.py` 新增 `RESIDUAL_TYPED = 3` 路由常量
- `ResidualSegment` 新增 `residual_type` 字段
- 新增 typed residual 编码/解码：enum / version / path_or_url / number_with_unit / identifier
- `router.py` 接受 `record_groups` 参数，新增 `_evaluate_record_group_route()` 族级路由评估
- 新增 `tests/test_segment.py`（10 个测试）
- 新增 benchmark matrix：6 类结构化样本 + `bench.py` runner + `test_benchmark_matrix.py`
- 154 个测试全部通过

### v0.3.5 — code quality + robustness hardening

- `_merge_priors()` 浮点精度阈值修复（`> 1e-15` 替代 `> 0`，避免静默移除字符）
- `_match_key_value()` 嵌套括号安全处理（拒绝嵌套括号的 suffix 提取）
- structured 路径前 `_should_skip_structured_api()` 轻量启发式预检（小文本/低结构化文本跳过 API）
- `probs_to_cdf()` redistribution 极端情况兜底（diff 残留强制平衡）
- `_looks_like_config()` 行级模式检测修复（避免误判包含多个中文冒号的 prose）
- `router.py` `detect_template()` 预扫描缓存（消除每段调用两次的冗余）
- `_count_all_phrase_occurrences()` 单次全文扫描优化（替代 O(n*m*k) 独立扫描）
- 内部函数 `_get_priors` / `_structured_online_compress` 重命名为公共 API
- `Header.version` 默认值与 `VERSION` 常量同步
- 137 个测试全部通过

### v0.3.4 — schema-seeded typed slot structured online

- `ApiClient.analyze_text()` 现在可返回 `document_family` / `block_families` / `field_schemas` / `slot_hints` / `enum_candidates`
- `online_manifest.py` 现在可持久化 schema/slot 级提示，并保持旧 payload 兼容
- `template_codec.py` 现在支持 typed slot payload：
  - `identifier`
  - `version`
  - `path_or_url`
  - `enum`
  - `number_with_unit`
- template route 现在会把 typed slot payload 作为真实成本参与收益比较
- `router.py` / `stats` 现在会记录 typed slot / typed template / template family 统计
- `zippedtext info` / `bench` 现在可直接查看 typed slot 命中与 template family 分布
- `bench` headline 结果改为走 public structured path，同时保留 raw structured diagnostic
- 新增 CLI 测试与结构化样本 fixture

## 常见问题

### Q: 为什么不继续把 legacy online 当主线优化？

因为 prediction cache 本质上仍然是在存“模型输出文本”，而不是压缩真正需要的最小结构化信息。

### Q: structured online 为什么更符合长期方向？

因为它让 LLM 参与：
- 结构分析
- 分段
- gain-based 路由
- 短语/术语发现
- template codec
- residual 架构
- side info 设计
- schema / slot hints

而不是只做 fragile 的 next-token 预测。

### Q: structured online 解压为什么不需要 API？

因为 LLM 只参与编码期建模；解压期只依赖 `.ztxt` v3 中的结构化 side info 与本地确定性 coder。

### Q: 为什么有时 `--mode online --sub-mode structured` 最后还是得到 offline 文件？

因为 `compress()` 仍然会比较 whole-file 最终收益；如果 structured 结果不如 offline，小文件或结构收益不足的样本会自动回退 offline。这是预期行为，不是错误。

### Q: 能否压缩二进制文件？

不能。当前设计专为 Unicode 纯文本优化。

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
