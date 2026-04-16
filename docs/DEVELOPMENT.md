# LLMChineseCompression — 开发指南

> 本文档面向项目维护者和后续 AI Agent，包含项目总览、测试与发布流程、当前版本状态和中期路线图。

---

## 1. 项目总览

### 基本信息

| 项目 | 值 |
|------|-----|
| 仓库 | `Roast-2007/LLMChineseCompression` |
| 包名 | `zippedtext` |
| 当前版本 | `0.3.6` |
| Python | >= 3.12 |
| 协议 | MIT |
| 构建后端 | hatchling |
| CLI 入口 | `zippedtext` |

### 当前架构定位

项目当前维护两类 online 路径：

1. **structured online（默认）**
   - LLM 在压缩期参与 analysis、分段、route、术语/短语发现；
   - analysis manifest 已扩展到 document/schema/slot hints；
   - `.ztxt` v3 保存结构化 side info；
   - analysis manifest 中的字符先验会并入 structured coder；
   - `list` / `table` / `config` 会优先按行切分，提高 template route 命中率；
   - template slot 已开始支持 typed slot codec；
   - 解压不依赖远端 API。

2. **legacy online（兼容）**
   - `char` / `token` 子模式；
   - 保留 prediction cache 路径，用于兼容、对照和回归测试；
   - 不再作为主架构继续扩展。

### 核心架构

```text
用户文本
  │
  ▼
┌────────────────────────────────────────────────────────────┐
│ compressor.py — 压缩/解压编排器                             │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ encoder.py   │  │ decoder.py   │  │ codegen.py   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                  │                                │
│  ┌──────▼──────────────────▼──────────────────────────────┐ │
│  │ predictor/                                             │ │
│  │  adaptive.py   PPM 核心                                │ │
│  │  priors.py     中文字频先验                            │ │
│  │  phrases.py    短语级编码                              │ │
│  │  llm.py        legacy online 预测器                    │ │
│  └──────┬─────────────────────────────────────────────────┘ │
│         │                                                   │
│  ┌──────▼────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ online_manifest.py│  │ segment.py   │  │ router.py    │ │
│  │ structured 元数据 │  │ 分段/分类     │  │ 路由/收益估计 │ │
│  └──────┬────────────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                      │                  │         │
│  ┌──────▼────────────┐  ┌──────▼────────────┐  ┌──────────▼─────────┐
│  │ term_dictionary.py│  │ template_codec.py │  │ api_client.py       │
│  │ 术语/短语字典生成  │  │ 模板/typed slot    │  │ LLM 分析接口         │
│  └───────────────────┘  └───────────────────┘  └────────────────────┘
└──────────────────────────┬────────────────────────────────────────────┘
                           ▼
                  format.py → .ztxt (v1 / v2 / v3)
```

### 关键设计决策

1. **无静态词表**：使用逃逸编码动态构建词汇。
2. **PPM 风格自适应**：order-0 到 order-6（可配置）的字符级 n-gram 模型。
3. **structured online 只参与编码期**：LLM 负责建模与结构提示，解码期不依赖远端 API。
4. **legacy online 冻结**：`char` / `token` 继续保留，但不再作为主架构继续扩展。
5. **section 化 side info**：v3 用 typed sections 表达 analysis / dictionary / templates / segment metadata / route stats。
6. **版本号在两处维护**：`__init__.py` 和 `pyproject.toml`，发版时必须同步。
7. **whole-file fallback 保留**：`compress()` 发现 structured 结果不如 offline 时，仍然会自动回退 offline。

---

## 2. `.ztxt` 格式

### v2（legacy offline / legacy online / codegen）

```text
Offset  Size  Field
0       4     Magic (b'ZTXT')
4       1     Version (0x02)
5       1     Mode (0x00=online, 0x01=offline, 0x02=codegen)
6       1     Flags (bit0: phrases, bit1: priors)
7       1     Max PPM order (4/5/6)
8       4     Token count
12      4     Original byte length
16      4     CRC32
20      4     Model data length
24      4     Phrase table length
28      4     Reserved
32+     var   Model data / Phrase table / Compressed body
```

### v3（structured online）

```text
Offset  Size  Field
0       4     Magic (b'ZTXT')
4       1     Version (0x03)
5       1     Mode (0x00=online)
6       1     Flags
7       1     Max PPM order
8       4     Token count
12      4     Original byte length
16      4     CRC32
20      4     Metadata length
24      4     Payload length
28      4     Reserved
32+     var   Section metadata
...     var   Payload body
```

当前 v3 section 类型：
- `SECTION_ANALYSIS`
- `SECTION_PHRASE_TABLE`
- `SECTION_TEMPLATES`
- `SECTION_SEGMENTS`
- `SECTION_STATS`

analysis section 现在可增量携带：
- `document_family`
- `block_families`
- `field_schemas`
- `slot_hints`
- `enum_candidates`

section entry 自带 flags，可选择 raw / zstd codec；读取时会保留 section flags 与 stored size。

---

## 3. 版本发布流程

### 3.1 语义化版本

遵循 [SemVer](https://semver.org/lang/zh-CN/)：`主版本.次版本.修订号`

- **修订号**：bug 修复 / 行为修正
- **次版本**：新功能，向后兼容
- **主版本**：破坏性变更

### 3.2 发版步骤

1. 确认所有本地回归通过：`pytest tests/`
2. 使用真实 DeepSeek key 跑在线集成：`DEEPSEEK_API_KEY=... pytest tests/test_online_integration.py -v -s`
3. 使用结构化样本跑 `zippedtext bench` 与 `c/info/d` 端到端校验
4. 更新版本号：`src/zippedtext/__init__.py` + `pyproject.toml`
5. 更新 README / `docs/DEVELOPMENT.md` / roadmap 中的当前版本与正文状态
6. 提交：`git commit -m "release: v0.X.X — 简要描述"`
7. 打 tag：`git tag v0.X.X`
8. Push：`git push origin main && git push origin v0.X.X`
9. GitHub Release
10. （可选）PyPI：`python -m build && twine upload dist/*`

> 注意：真实 API key 只能通过环境变量或 CLI 参数传入；不要把 key 字面值写入仓库、文档或脚本。

---

## 4. AI Agent 开发须知

### 环境设置

```bash
git clone https://github.com/Roast-2007/LLMChineseCompression.git
cd LLMChineseCompression
pip install -e ".[dev]"
```

### 代码修改后的验证清单

- [ ] `pytest tests/` 全部通过
- [ ] 修改了 `adaptive.py` 或 `encoder.py` / `decoder.py`？→ 必须验证无损性
- [ ] 修改了 `format.py`？→ 确认 v1/v2/v3 兼容性
- [ ] 修改了 `arithmetic.py`？→ 必须用大词表（20000）测试
- [ ] 新增依赖？→ 更新 `pyproject.toml`
- [ ] 改了版本号？→ 两处同步
- [ ] 修改了 structured route / template / residual？→ 验证 route reason、template hit、typed slot 统计与 API-free 解压
- [ ] 改了 `bench` / `info`？→ 确认 public path 输出与 raw diagnostic 一致可解释

### 核心不变量（不可违反）

1. **编码/解码对称**：encoder.py 和 decoder.py 的每一步必须完全镜像
2. **CDF 确定性**：`probs_to_cdf()` 不能有任何非确定性行为
3. **预测器同步**：`AdaptivePredictor` 在编码器和解码器中的调用顺序完全一致
4. **CRC32 校验**：解压后必须验证，不通过则报错
5. **短语表同步**：encoder / decoder 必须以相同顺序添加短语到 predictor
6. **structured online 解压不访问 API**：新的 v3 online 文件必须只依赖本地 side info 完成解压
7. **whole-file fallback 仍然有效**：`compress()` 在 structured 结果不划算时仍可自动回退 offline
8. **priors 合并安全**：`_merge_priors()` 合并字符先验时不得因浮点精度静默移除字符（阈值 `> 1e-15`）

---

## 5. 测试与验证

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

当前重点测试文件：
- `tests/test_analysis_manifest.py`
- `tests/test_router.py`
- `tests/test_template_codec.py`
- `tests/test_residual.py`
- `tests/test_gain_estimator.py`
- `tests/test_online_structured.py`
- `tests/test_format_v3.py`
- `tests/test_cli.py`
- `tests/test_online_integration.py`

建议的结构化 bench 样本：
- `tests/sample_structured_api.txt`
- `tests/sample_structured_config.txt`

> 注：在当前 Windows + Python 3.14 环境下，`pytest` 结束阶段偶发 access violation 噪声，但新增与回归用例均已实际通过；问题更像解释器/环境级异常，而不是本次 structured online 逻辑错误。

---

## 6. 路线图

### v0.3.3 — structured online 第二阶段 ✅

- [x] template codec 最小闭环（key-value / list prefix / table row）
- [x] residual architecture（template residual 复用 literal / phrase coder）
- [x] online gain estimator（literal / phrase / template 净收益选路）
- [x] stronger structured side-info compression（section flags + raw/zstd codec + compact binary metadata）
- [x] `zippedtext info` / `bench` side-info 拆解与 route 可观测性
- [x] analysis priors 真正接入 structured coder
- [x] `list` / `table` / `config` 的 line-level segmentation
- [x] hint-aware template detection / catalog pruning / richer fallback reason
- [x] structured online API smoke test

### v0.3.4 — schema-seeded typed slot structured online ✅

- [x] `ApiClient.analyze_text()` 扩展到 document/schema/slot hints
- [x] `AnalysisManifest` 支持 `document_family` / `block_families` / `field_schemas` / `slot_hints` / `enum_candidates`
- [x] typed slot codec 初版：
  - [x] `identifier`
  - [x] `version`
  - [x] `path_or_url`
  - [x] `enum`
  - [x] `number_with_unit`
- [x] `router.py` / `stats` 记录 typed slot / typed template / template family 统计
- [x] `zippedtext info` / `bench` 展示 typed slot 与 family 观测值
- [x] `bench` headline 结果改为走 public structured path，并保留 raw structured diagnostic
- [x] CLI 测试 + 结构化 fixture

### v0.3.6 — record family expansion ✅

- [x] `RecordGroup` 数据结构（`kind` / `segment_indices` / `family` / `text_span`）
- [x] `group_record_groups()` 连续结构化段落聚类
- [x] `record` 模板 kind 支持多行记录模板
- [x] `TemplateCatalog` 序列化支持 JSON-encoded per-line skeleton
- [x] `_render_record_template()` 多行渲染
- [x] `_match_record_template()` 多行记录模板检测
- [x] typed residual（`RESIDUAL_TYPED`）：enum / version / path_or_url / number_with_unit / identifier
- [x] `router.py` 接受 `record_groups` 参数 + `_evaluate_record_group_route()`
- [x] `tests/test_segment.py` 10 个新增测试
- [x] benchmark matrix：6 类结构化样本 + `bench.py` + `test_benchmark_matrix.py`
- [x] 154 个测试全部通过

### v0.3.5 — code quality + robustness hardening ✅

- [x] `_merge_priors()` 浮点精度过滤修复（阈值改为 `> 1e-15`，避免静默移除字符）
- [x] `_match_key_value()` 嵌套括号歧义处理（拒绝嵌套括号 value 的 suffix 提取）
- [x] structured 路径前轻量启发式预检（`_should_skip_structured_api`，小文本/低结构化度文本跳过 API）
- [x] `probs_to_cdf()` redistribution 极端情况兜底（diff 残留强制平衡）
- [x] `_looks_like_config()` 误判修复（改为行级 `key=value` 模式检测）
- [x] `detect_template()` 重复调用优化（router.py 预扫描缓存，避免每段调用两次）
- [x] `_count_phrase_occurrences()` 单次全文扫描优化（`_count_all_phrase_occurrences`）
- [x] `_get_priors` / `_structured_online_compress` 重命名为公共 API（`get_priors` / `structured_compress`）
- [x] `Header.version` 默认值与 `VERSION` 常量同步（v3）
- [x] `compressor.py` 新增 `import re`（支持结构化预检）

### v0.3.6 — record family expansion ✅

- [x] record grouping
- [x] multi-line / paragraph template family
- [x] typed residual 第一轮
- [x] benchmark matrix 初版

### 后续仍未完成

- [ ] multi-line / record template family
- [ ] document-level family clustering
- [ ] global / family-level gain optimization 深化
- [ ] entity / alias / bilingual term system
- [ ] hierarchical residual
- [ ] benchmark matrix 完整报表
- [ ] MoE probability layer
- [ ] 本地确定性模型

---

## 7. 当前判断

`v0.3.6` 之后，structured online 的下一阶段重点已经更清晰：
- 不应回头深挖 legacy char/token
- 不应只继续做 prompt 微调
- 更高 ROI 的方向是：
  - multi-line / record template
  - document-level family clustering
  - typed residual 与更强的 family-level amortization
  - entity / alias / bilingual term system

一句话总结：

> 继续让 LLM 帮压缩器找到“更短的可逆结构表示”，比让 LLM 多猜几个 next token 更重要。

---

## 8. 常见问题

### Q: 为什么 structured online 仍可能最终回退到 offline？

因为 `compress()` 仍然会做 whole-file 最终比较；如果 structured 结果不如 offline，就会自动回退。这是预期行为。

### Q: 为什么 `bench` 现在同时显示 structured 和 raw diagnostic？

因为用户真正拿到的是 public `compress(...sub_mode="structured")` 的最终结果，但调优时还需要看到 whole-file fallback 前的 raw structured 表现，两者都需要可观测。

### Q: 下一步最值得做什么？

优先做：
1. multi-line / record template family
2. document-level family clustering + global gain optimization
3. typed residual / entity-alias-term system
4. 更完整 benchmark matrix

### Q: structured online 解压为什么仍然不需要 API？

因为 LLM 只参与编码期建模；解压期只依赖 `.ztxt` v3 中的 side info 与本地确定性 coder。

---

## 9. 代码审查记录（v0.3.5 修复）

### 2026-04-12 全项目代码审查修复

| 编号 | 严重度 | 模块 | 问题 | 修复 |
|------|--------|------|------|------|
| S1 | 严重 | `compressor.py` | `_merge_priors` 的 `value > 0` 过滤可能因浮点精度静默移除字符 | 改为 `> 1e-15` 阈值 |
| S2 | 严重 | `template_codec.py` | `_match_key_value` 的 `rsplit("（", 1)` 对嵌套括号会误切分 | 增加嵌套检测，拒绝嵌套括号的 suffix 提取 |
| S3 | 严重 | `compressor.py` | 小文本/低结构化文本仍调用 LLM API 产生不必要费用 | 新增 `_should_skip_structured_api()` 轻量启发式预检 |
| M1 | 中等 | `arithmetic.py` | `probs_to_cdf()` redistribution 极端情况下 diff 可能残留 | 增加 diff 兜底和最终强制平衡逻辑 |
| M3 | 中等 | `segment.py` | `_looks_like_config` 用 `text.count(":") >= 3` 误判 prose | 改为行级 `key=value` 模式正则检测 |
| M6 | 中等 | `compressor.py` | `_get_priors` / `_structured_online_compress` 被 cli.py 直接导入，破坏封装 | 重命名为 `get_priors` / `structured_compress` |
| L1 | 轻微 | `format.py` | `Header.version` 默认值 `VERSION_V2` 与 `VERSION = VERSION_V3` 不同步 | 改为 `version: int = VERSION` |
| L2 | 轻微 | `router.py` | `detect_template()` 对每个 segment 调用两次 | 预扫描缓存，主循环复用 |
| L3 | 轻微 | `term_dictionary.py` | `_count_phrase_occurrences` 对每个短语独立扫描全文 O(n*m*k) | 新增 `_count_all_phrase_occurrences` 单次扫描 |
