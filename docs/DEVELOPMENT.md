# LLMChineseCompression — 开发指南

> 本文档面向项目维护者和后续 AI Agent，包含项目总览、测试与发布流程、当前版本状态和中期路线图。

---

## 1. 项目总览

### 基本信息

| 项目 | 值 |
|------|-----|
| 仓库 | `Roast-2007/LLMChineseCompression` |
| 包名 | `zippedtext` |
| 当前版本 | `0.3.3` |
| Python | >= 3.12 |
| 协议 | MIT |
| 构建后端 | hatchling |
| CLI 入口 | `zippedtext` |

### 当前架构定位

项目当前维护两类 online 路径：

1. **structured online（默认）**
   - LLM 在压缩期参与 analysis、分段、route、术语/短语发现；
   - `.ztxt` v3 保存结构化 side info；
   - analysis manifest 中的字符先验会并入 structured coder；
   - `list` / `table` / `config` 会优先按行切分，提高 template route 命中率；
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
│  │ 术语/短语字典生成  │  │ 模板/残差编码      │  │ LLM 分析接口         │
│  └───────────────────┘  └───────────────────┘  └────────────────────┘
└──────────────────────────┬────────────────────────────────────────────┘
                           ▼
                  format.py → .ztxt (v1 / v2 / v3)
```

### 文件清单与职责

```text
src/zippedtext/
├── __init__.py                # 版本号
├── __main__.py                # python -m zippedtext 入口
├── cli.py                     # CLI 命令定义
├── cli_config.py              # config 子命令组
├── config.py                  # 配置系统
├── provider.py                # API 平台预设
├── compressor.py              # 核心：compress() / decompress()
├── encoder.py                 # 编码逻辑
├── decoder.py                 # 解码逻辑
├── cdf_utils.py               # uniform CDF 工具
├── codegen.py                 # 代码生成模式
├── arithmetic.py              # 32-bit 整数算术编码器/解码器
├── bitstream.py               # BitOutputStream / BitInputStream
├── format.py                  # .ztxt 格式 v1 + v2 + v3
├── api_client.py              # ApiClient（OpenAI 协议兼容）
├── online_manifest.py         # structured online 元数据与统计
├── sideinfo_codec.py          # v3 section codec / side info helper
├── template_codec.py          # template catalog / payload codec
├── residual.py                # residual 编码与解码辅助
├── gain_estimator.py          # segment gain estimator
├── segment.py                 # 文本分段与段类型分类
├── router.py                  # segment route / 收益判断
├── term_dictionary.py         # LLM + heuristic 短语/术语字典
└── predictor/
    ├── base.py                # Predictor 抽象基类
    ├── adaptive.py            # ★ 核心：自适应 PPM 预测器
    ├── priors.py              # 中文字符频率先验
    ├── phrases.py             # 短语级编码
    ├── llm.py                 # legacy online 预测器
    └── ngram.py               # N-gram 预测器
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

section entry 自带 flags，可选择 raw / zstd codec；读取时会保留 section flags 与 stored size。

---

## 3. 版本发布流程

### 3.1 语义化版本

遵循 [SemVer](https://semver.org/lang/zh-CN/)：`主版本.次版本.修订号`

- **修订号**：bug 修复 / 行为修正
- **次版本**：新功能，向后兼容
- **主版本**：破坏性变更

### 3.2 发版步骤

1. 确认所有测试通过：`pytest tests/`
2. 更新版本号：`src/zippedtext/__init__.py` + `pyproject.toml`
3. 更新 README / `docs/DEVELOPMENT.md` / roadmap 中的当前版本与正文状态
4. 提交：`git commit -m "release: v0.X.X — 简要描述"`
5. 打 tag：`git tag v0.X.X`
6. Push：`git push origin main && git push origin v0.X.X`
7. GitHub Release
8. （可选）PyPI：`python -m build && twine upload dist/*`

---

## 4. 用户如何同步更新

```bash
pip install --upgrade git+https://github.com/Roast-2007/LLMChineseCompression.git
```

---

## 5. AI Agent 开发须知

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
- [ ] 修改了 structured route / template / residual？→ 验证 route reason、template hit 与 API-free 解压

### 核心不变量（不可违反）

1. **编码/解码对称**：encoder.py 和 decoder.py 的每一步必须完全镜像
2. **CDF 确定性**：`probs_to_cdf()` 不能有任何非确定性行为
3. **预测器同步**：`AdaptivePredictor` 在编码器和解码器中的调用顺序完全一致
4. **CRC32 校验**：解压后必须验证，不通过则报错
5. **短语表同步**：encoder / decoder 必须以相同顺序添加短语到 predictor
6. **structured online 解压不访问 API**：新的 v3 online 文件必须只依赖本地 side info 完成解压
7. **whole-file fallback 仍然有效**：`compress()` 在 structured 结果不划算时仍可自动回退 offline

---

## 6. 测试与验证

```bash
# 全部测试
pytest tests/

# structured online 重点回归
pytest tests/test_analysis_manifest.py tests/test_router.py tests/test_template_codec.py tests/test_residual.py tests/test_gain_estimator.py tests/test_online_structured.py tests/test_format_v3.py

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
- `tests/test_online_integration.py`

> 注：在当前 Windows + Python 3.14 环境下，`pytest` 结束阶段偶发 access violation 噪声，但新增与回归用例均已实际通过；问题更像解释器/环境级异常，而不是本次 structured online 逻辑错误。

---

## 7. 路线图

### v0.2.0 — DeepSeek API 在线模式 ✅

- [x] online 模式（char + token 子模式）
- [x] API 确定性保证
- [x] 在线模式测试

### v0.3.0 — 压缩率优化 + 多平台支持 ✅

- [x] 中文字符频率先验
- [x] 短语级编码
- [x] order-5 / order-6 实验
- [x] 代码生成模式
- [x] 多平台 API 支持
- [x] 交互式配置系统
- [x] `.ztxt` v2

### v0.3.1 — legacy online 性能优化 ✅

- [x] `CHUNK_CHARS` 20 → 200
- [x] prediction cache 嵌入 `.ztxt`
- [x] 在线文件智能回退 offline
- [x] model_data 存储 chunk_chars / max_tokens / prediction_cache
- [x] 修复 DeepSeek API 长上下文非确定性问题

### v0.3.2 — structured online 初步落地 ✅

- [x] 冻结 `online-legacy-char`
- [x] 冻结 `online-legacy-token`
- [x] structured online 设为默认 online 子模式
- [x] `ApiClient.analyze_text()` 返回结构化 manifest
- [x] 引入 `segment.py`
- [x] 引入 `router.py`
- [x] 引入 `term_dictionary.py`
- [x] 引入 `.ztxt` v3
- [x] 引入 structured online 测试
- [x] structured online API-free 解压

### v0.3.3 — structured online 第二阶段 ✅

- [x] template codec 最小闭环（key-value / list prefix / table row）
- [x] residual architecture（template residual 复用 literal / phrase coder）
- [x] online gain estimator（literal / phrase / template 净收益选路）
- [x] stronger structured side-info compression（section flags + raw/zstd codec + compact binary metadata）
- [x] `zippedtext info` / `bench` side-info 拆解与 route 可观测性
- [x] analysis priors 真正接入 structured coder
- [x] `list` / `table` / `config` 的 line-level segmentation
- [x] hint-aware template detection / catalog pruning / richer fallback reason
- [x] `tests/test_template_codec.py`
- [x] `tests/test_residual.py`
- [x] `tests/test_gain_estimator.py`
- [x] structured online API smoke test

### 后续仍未完成

- [ ] 更完整模板体系（跨行列表、文档段模板、更多配置模板）
- [ ] benchmark matrix 完整报表
- [ ] mixture-of-experts probability layer
- [ ] 本地确定性模型
- [ ] Rust 核心（PyO3）

---

## 8. 常见问题

### Q: 为什么不继续把 legacy online 当主线优化？

因为 prediction cache 本质上仍然是在存“模型输出文本”，而不是存压缩真正需要的最小结构化信息。

### Q: structured online 为什么更符合长期方向？

因为它让 LLM 参与：
- 结构分析
- 分段
- gain-based 路由
- 短语/术语发现
- template codec
- residual 架构
- side info 设计

而不是只做 fragile 的 next-token 预测。

### Q: structured online 解压为什么不需要 API？

因为 LLM 只参与编码期建模；解压期只依赖 `.ztxt` v3 中的结构化 side info 与本地确定性 coder。

### Q: 为什么有时 `--mode online --sub-mode structured` 最后还是得到 offline 文件？

因为 `compress()` 仍然会比较 whole-file 最终收益；如果 structured 结果不如 offline，小文件或结构收益不足的样本会自动回退 offline。这是预期行为，不是错误。

### Q: 能否压缩二进制文件？

不能。当前设计专为 Unicode 纯文本优化。
