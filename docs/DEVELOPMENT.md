# LLMChineseCompression — 开发指南

> 本文档面向项目维护者和后续 AI Agent，包含项目总览、版本发布流程、用户同步方式和后续路线图。

---

## 1. 项目总览

### 基本信息

| 项目 | 值 |
|------|-----|
| 仓库 | `Roast-2007/LLMChineseCompression` |
| 包名 | `zippedtext` |
| 版本 | 见 `src/zippedtext/__init__.py` 和 `pyproject.toml` 中的 `version` 字段 |
| Python | >= 3.12 |
| 协议 | MIT |
| 构建后端 | hatchling |
| CLI 入口 | `zippedtext` (通过 `pyproject.toml` 的 `[project.scripts]`) |

### 核心架构

```
用户文本
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ compressor.py — 压缩/解压编排器                      │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ encoder.py   │  │ decoder.py   │  │ codegen.py│  │
│  │ 编码逻辑      │  │ 解码逻辑      │  │ 代码生成  │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘  │
│         │                  │                         │
│  ┌──────▼──────────────────▼───────┐                 │
│  │ predictor/                       │                │
│  │  adaptive.py  PPM 预测器 (核心)   │                │
│  │  priors.py    中文字频先验        │                │
│  │  phrases.py   短语级编码          │                │
│  │  llm.py       LLM 在线预测器     │                │
│  └──────┬──────────────────────────┘                 │
│         │                                            │
│  ┌──────▼───────┐  ┌──────────────┐  ┌───────────┐  │
│  │ arithmetic.py│  │ bitstream.py │  │cdf_utils.py│ │
│  │ 算术编码器    │  │ 比特级 I/O   │  │ CDF 工具   │ │
│  └──────────────┘  └──────────────┘  └───────────┘  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ config.py    │  │ provider.py  │  │api_client.py│ │
│  │ 配置系统      │  │ 平台预设     │  │ API 客户端  │ │
│  └──────────────┘  └──────────────┘  └───────────┘  │
└──────────────────────────┬──────────────────────────┘
                           ▼
                    format.py → .ztxt 文件 (v1/v2)
```

### 文件清单与职责

```
LLMChineseCompression/
├── pyproject.toml                 # 构建配置、依赖、PyPI 元数据
├── README.md                      # 用户文档（中文）
├── LICENSE                        # MIT 协议
├── .gitignore                     # Git 忽略规则
├── docs/
│   └── DEVELOPMENT.md             # 本文档：开发指南
├── src/zippedtext/
│   ├── __init__.py                # 版本号（__version__）
│   ├── __main__.py                # python -m zippedtext 入口
│   ├── cli.py                     # CLI 命令定义（click）
│   │                              #   c — 压缩  d — 解压
│   │                              #   info — 文件信息  bench — 基准测试
│   ├── cli_config.py              # config 子命令组 (v0.3.0)
│   │                              #   init / show / set / models
│   ├── config.py                  # 配置系统 (v0.3.0)
│   ├── provider.py                # API 平台预设 (v0.3.0)
│   ├── compressor.py              # 核心：compress() / decompress()
│   ├── encoder.py                 # 编码逻辑（从 compressor.py 拆分）
│   ├── decoder.py                 # 解码逻辑（从 compressor.py 拆分）
│   ├── cdf_utils.py               # uniform CDF 工具
│   ├── codegen.py                 # 代码生成模式 (v0.3.0)
│   ├── arithmetic.py              # 32-bit 整数算术编码器/解码器
│   ├── bitstream.py               # BitOutputStream / BitInputStream
│   ├── format.py                  # .ztxt 格式 v1 + v2
│   ├── api_client.py              # ApiClient（OpenAI 协议兼容）
│   ├── tokenizer.py               # 字符级 Vocab
│   └── predictor/
│       ├── __init__.py            # 导出
│       ├── base.py                # Predictor 抽象基类
│       ├── adaptive.py            # ★ 核心：自适应 PPM 预测器
│       │                          #   动态词汇 + 逃逸编码 + 可配置 order
│       │                          #   支持 priors + phrases
│       ├── priors.py              # 中文字符频率先验 (v0.3.0)
│       ├── phrases.py             # 短语级编码 (v0.3.0)
│       ├── llm.py                 # LLM 在线预测器
│       └── ngram.py               # N-gram 预测器
└── tests/
    ├── test_arithmetic.py         # 算术编码单元测试
    ├── test_roundtrip.py          # 端到端无损测试
    ├── test_online.py             # 在线模式 mock 测试
    ├── test_online_integration.py # 在线模式 API 集成测试
    ├── test_format_v2.py          # 格式 v2 测试 (v0.3.0)
    ├── test_config.py             # 配置系统测试 (v0.3.0)
    ├── test_phrases.py            # 短语编码测试 (v0.3.0)
    ├── test_codegen.py            # 代码生成测试 (v0.3.0)
    ├── sample_cn.txt              # 中文测试语料
    └── sample_large.txt           # 中英混合测试语料
```

### 关键设计决策

1. **无静态词表**：使用逃逸编码动态构建词汇。不需要在压缩文件中存储任何词表或模型。
2. **PPM 风格自适应**：order-0 到 order-6（可配置）的字符级 n-gram 模型。
3. **中文频率先验**（v0.3.0）：top-3000 字符频率作为 warm start，改善短文本。
4. **短语级编码**（v0.3.0）：高频短语（2-8 字符）作为单一符号。
5. **32-bit 整数算术编码**：避免浮点精度问题，保证无损。
6. **多平台 API**（v0.3.0）：支持任何 OpenAI 协议兼容 API，配置持久化到本地。
7. **代码生成模式**（v0.3.0）：LLM 识别可用代码表示的文本片段，实验性。
8. **版本号在两处维护**：`__init__.py` 和 `pyproject.toml`，发版时必须同步。

### .ztxt 格式 v2（v0.3.0）

v0.3.0 引入 32 字节的 v2 header，向后兼容 v1：

```
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

`read_file()` 按 version 字节分发：v0x01 → 旧 24 字节 header，v0x02 → 新 32 字节 header。

---

## 2. 版本发布流程

### 2.1 语义化版本

遵循 [SemVer](https://semver.org/lang/zh-CN/)：`主版本.次版本.修订号`

- **修订号**（0.1.0 → 0.1.1）：bug 修复，无 API 变更
- **次版本**（0.1.x → 0.2.0）：新功能，向后兼容
- **主版本**（0.x.x → 1.0.0）：破坏性变更

### 2.2 发版步骤

1. 确认所有测试通过：`pytest tests/`
2. 更新版本号：`src/zippedtext/__init__.py` + `pyproject.toml`
3. 更新 README.md 的 CHANGELOG
4. 提交：`git commit -m "release: v0.X.X — 简要描述"`
5. 打 tag：`git tag v0.X.X`
6. Push：`git push origin main && git push origin v0.X.X`
7. GitHub Release
8. （可选）PyPI：`python -m build && twine upload dist/*`

---

## 3. 用户如何同步更新

```bash
# 从 GitHub
pip install git+https://github.com/Roast-2007/LLMChineseCompression.git

# 升级
pip install --upgrade git+https://github.com/Roast-2007/LLMChineseCompression.git
```

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
- [ ] 修改了 `adaptive.py` 或 `encoder.py`/`decoder.py`？→ 必须验证无损性
- [ ] 修改了 `format.py`？→ 确认 v1/v2 兼容性
- [ ] 修改了 `arithmetic.py`？→ 必须用大词表（20000）测试
- [ ] 新增依赖？→ 更新 `pyproject.toml`
- [ ] 改了版本号？→ 两处同步

### 核心不变量（不可违反）

1. **编码/解码对称**：encoder.py 和 decoder.py 的每一步必须完全镜像
2. **CDF 确定性**：`probs_to_cdf()` 不能有任何非确定性行为
3. **预测器同步**：`AdaptivePredictor` 在编码器和解码器中的调用顺序完全一致
4. **CRC32 校验**：解压后必须验证，不通过则报错
5. **短语表同步**：encoder/decoder 必须以相同顺序添加短语到 predictor

---

## 5. 路线图

### v0.2.0 — DeepSeek API 在线模式 ✅

- [x] online 模式（char + token 子模式）
- [x] API 确定性保证
- [x] 在线模式测试

### v0.3.0 — 压缩率优化 + 多平台支持 ✅

- [x] 预置中文字符频率表（top-3000）
- [x] 短语级编码（2-8 字符高频短语）
- [x] order-5/order-6 实验
- [x] 代码生成模式（实验性）
- [x] 多平台 API 支持（SiliconFlow 等）
- [x] 交互式配置系统
- [x] .ztxt 格式 v2

### v0.3.1 — 在线模式性能优化 ✅

- [x] `CHUNK_CHARS` 20 → 200，API 调用减少 10 倍
- [x] 预测缓存嵌入 .ztxt 文件，解压无需 API（~1s）
- [x] 智能回退：在线+缓存文件若大于离线则自动使用离线
- [x] model_data 存储 chunk_chars、max_tokens、prediction_cache
- [x] 修复 DeepSeek API 长上下文非确定性问题

### v0.4.0 — 性能优化

- [ ] Rust 核心（PyO3）：`arithmetic.py` + `bitstream.py` 用 Rust 重写
- [ ] 流式压缩/解压
- [ ] 并行压缩

### v0.5.0 — 工具链完善

- [ ] PyPI 发布
- [ ] 命令行自动补全
- [ ] `zippedtext convert` — 批量压缩
- [ ] `zippedtext diff` — 比较 .ztxt 文件
- [ ] 压缩率可视化

### v1.0.0 — 稳定版

- [ ] 完整 API 文档
- [ ] 80%+ 测试覆盖率
- [ ] 跨平台 CI
- [ ] 性能回归测试

---

## 6. 常见问题

### Q: 为什么不用 BPE tokenizer？

字符级方案保证完美可逆性和零外部依赖。BPE 需要存储 tokenizer 配置且可能有 UNK token。

### Q: 为什么短文本压缩率不好？

每个新字符首次出现约需 17-20 bits（CJK）。v0.3.0 的中文频率先验大幅改善此问题。

### Q: 能否压缩二进制文件？

不能。当前设计专为 Unicode 纯文本优化。
