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
┌─────────────────────────────────────────────┐
│ compressor.py — 压缩/解压编排器              │
│                                              │
│  ┌──────────────┐    ┌────────────────────┐  │
│  │ adaptive.py  │───▶│ arithmetic.py      │  │
│  │ PPM 预测器    │    │ 算术编码器/解码器    │  │
│  │ (概率分布)    │    │ (比特流输出)        │  │
│  └──────────────┘    └────────┬───────────┘  │
│                               │              │
│  ┌──────────────┐    ┌────────▼───────────┐  │
│  │ api_client.py│    │ bitstream.py       │  │
│  │ DeepSeek API │    │ 比特级 I/O         │  │
│  │ (可选增强)    │    └────────┬───────────┘  │
│  └──────────────┘             │              │
└───────────────────────────────┼──────────────┘
                                ▼
                         format.py → .ztxt 文件
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
│   │                              #   c — 压缩
│   │                              #   d — 解压
│   │                              #   info — 文件信息
│   │                              #   bench — 基准测试
│   ├── compressor.py              # 核心：compress() / decompress()
│   │                              # 逃逸编码动态词汇 + 自适应算术编码
│   ├── arithmetic.py              # 32-bit 整数算术编码器/解码器
│   │                              # probs_to_cdf() — 概率 → 整数 CDF
│   ├── bitstream.py               # BitOutputStream / BitInputStream
│   ├── format.py                  # .ztxt 二进制格式 v1（24B header）
│   ├── api_client.py              # DeepSeekClient（OpenAI 兼容）
│   ├── tokenizer.py               # 字符级 Vocab（当前未被核心流程使用）
│   └── predictor/
│       ├── __init__.py            # 导出
│       ├── base.py                # Predictor 抽象基类
│       ├── adaptive.py            # ★ 核心：自适应 PPM 预测器
│       │                          #   动态词汇 + 逃逸编码 + order-4 上下文
│       ├── llm.py                 # ★ LLM 在线预测器 (v0.2.0)
│       │                          #   LlmCharPredictor + LlmTokenPredictor
│       └── ngram.py               # N-gram 预测器（备用/离线增强）
└── tests/
    ├── test_arithmetic.py         # 算术编码单元测试
    ├── test_roundtrip.py          # 端到端无损测试
    ├── test_online.py             # 在线模式 mock 单元测试 (v0.2.0)
    ├── test_online_integration.py # 在线模式 API 集成测试 (v0.2.0)
    ├── sample_cn.txt              # 中文测试语料（462 字）
    └── sample_large.txt           # 中英混合测试语料（2622 字）
```

### 关键设计决策

1. **无静态词表**：使用逃逸编码动态构建词汇，新字符首次出现编码 `ESCAPE + Unicode codepoint`，后续直接由自适应模型编码。不需要在压缩文件中存储任何词表或模型。

2. **PPM 风格自适应**：order-0 到 order-4 的字符级 n-gram 模型，编码和解码过程中同步更新，保证确定性。

3. **32-bit 整数算术编码**：避免浮点精度问题，保证无损。CDF 总和精确等于 `2^32`。

4. **Unicode codepoint 分范围编码**：ASCII (7-bit) / CJK (15-bit) / Other (21-bit)，对中英文都高效。

5. **版本号在两处维护**：`src/zippedtext/__init__.py` 的 `__version__` 和 `pyproject.toml` 的 `version`。发版时两处必须同步。

---

## 2. 版本发布流程

### 2.1 语义化版本

遵循 [SemVer](https://semver.org/lang/zh-CN/)：`主版本.次版本.修订号`

- **修订号**（0.1.0 → 0.1.1）：bug 修复，无 API 变更
- **次版本**（0.1.x → 0.2.0）：新功能，向后兼容
- **主版本**（0.x.x → 1.0.0）：破坏性变更（如 .ztxt 格式不兼容）

### 2.2 发版步骤（逐步操作）

#### 步骤 1：确认所有测试通过

```bash
cd LLMChineseCompression
PYTHONPATH=src python tests/test_arithmetic.py
PYTHONPATH=src python tests/test_roundtrip.py
```

如果使用 pytest：
```bash
pip install pytest
pytest tests/
```

#### 步骤 2：更新版本号（两处）

**文件 1：`src/zippedtext/__init__.py`**
```python
__version__ = "0.2.0"  # 改为新版本号
```

**文件 2：`pyproject.toml`**
```toml
version = "0.2.0"  # 改为相同的新版本号
```

> 两处版本号必须完全一致。

#### 步骤 3：更新 CHANGELOG（如有）

在 README.md 或 CHANGELOG.md 中记录本次变更。

#### 步骤 4：提交并打 tag

```bash
git add -A
git commit -m "release: v0.2.0 — 简要描述变更"
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

#### 步骤 5：在 GitHub 创建 Release

1. 进入 https://github.com/Roast-2007/LLMChineseCompression/releases
2. 点击 **Draft a new release**
3. 选择刚推送的 tag（如 `v0.2.0`）
4. 填写标题和变更说明
5. 点击 **Publish release**

#### 步骤 6（可选）：发布到 PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

首次发布需要在 https://pypi.org 注册账号并配置 token。

---

## 3. 用户如何同步更新

### 3.1 从 GitHub 安装的用户

```bash
# 拉取最新代码
cd LLMChineseCompression
git pull origin main

# 重新安装
pip install .

# 或开发模式
pip install -e .
```

### 3.2 直接 pip 安装的用户（如果已发布到 PyPI）

```bash
pip install --upgrade zippedtext
```

### 3.3 一键安装最新版（无需 clone）

```bash
pip install git+https://github.com/Roast-2007/LLMChineseCompression.git
```

升级：
```bash
pip install --upgrade git+https://github.com/Roast-2007/LLMChineseCompression.git
```

---

## 4. .ztxt 格式兼容性注意事项

`.ztxt` 文件 header 包含 `version` 字段（当前 `0x01`）。**格式版本与软件版本独立**。

| 场景 | 处理方式 |
|------|---------|
| 新功能但格式不变 | 软件版本升级，格式版本不变，老文件兼容 |
| 格式有变但可兼容 | 格式版本不变，代码里做向后兼容处理 |
| 格式有破坏性变更 | 格式版本号 +1（如 `0x02`），老文件无法被新版解压时报明确错误 |

**规则**：尽量保持格式向后兼容。如果必须破坏兼容，提供迁移工具或在 Release Notes 中明确说明。

---

## 5. AI Agent 开发须知

### 环境设置

```bash
git clone https://github.com/Roast-2007/LLMChineseCompression.git
cd LLMChineseCompression
pip install -e ".[dev]"
```

### 运行测试

```bash
# 快速验证
PYTHONPATH=src python tests/test_arithmetic.py
PYTHONPATH=src python tests/test_roundtrip.py

# 完整 pytest
pytest tests/ -v
```

### 代码修改后的验证清单

- [ ] `test_arithmetic.py` 全部通过（算术编码往返）
- [ ] `test_roundtrip.py` 全部通过（端到端无损）
- [ ] 修改了 `adaptive.py` 或 `compressor.py`？→ 必须验证无损性
- [ ] 修改了 `format.py`？→ 确认格式版本兼容性
- [ ] 修改了 `arithmetic.py`？→ 必须用大词表（20000）测试
- [ ] 新增依赖？→ 更新 `pyproject.toml` 的 `dependencies`
- [ ] 改了版本号？→ 两处同步（`__init__.py` + `pyproject.toml`）

### 编码风格

- 类型注解：所有函数签名必须带类型标注
- 不可变优先：使用 `frozen=True` 的 dataclass
- 小文件：单文件不超过 400 行
- 错误处理：边界处验证，内部代码信任框架保证

### 核心不变量（不可违反）

1. **编码/解码对称**：`compressor._encode()` 和 `compressor._decode()` 的每一步必须完全镜像。任何打破对称的修改都会导致解压失败。
2. **CDF 确定性**：相同的概率输入必须产生相同的整数 CDF。`probs_to_cdf()` 不能有任何非确定性行为。
3. **预测器同步**：`AdaptivePredictor` 的 `predict()` 和 `update()` 在编码器和解码器中的调用顺序必须完全一致。
4. **CRC32 校验**：解压后必须验证 CRC32，不通过则报错，绝不静默返回错误数据。

---

## 6. 路线图

### v0.2.0 — DeepSeek API 在线模式 ✅ (已完成)

- [x] 实现 online 模式压缩/解压
  - 字符级子模式 (`--sub-mode char`，默认)：LLM 预测增强 PPM 概率分布
  - token 级子模式 (`--sub-mode token`，实验性)：API token 匹配 + 字符级回退
  - 分块处理：每 20 字符调用一次 API，获取续写预测
  - stop-on-mismatch 策略：预测偏离后自动停止增强，避免编码膨胀
- [x] 处理 API 确定性问题
  - `temperature=0` + `seed=42` 固定生成
  - 存储模型名称和子模式到 model_data 区段
  - 解压时需要 API 访问以重现相同预测
- [x] 在线模式压缩率：中文 ratio 0.53（对比离线 0.54，提升 ~2%）
  - 注：受 Chat API logprobs 精度限制（temperature=0 下概率完全peaked），提升幅度有限
  - 未来通过本地模型推理（v0.4.0+）可实现 ratio < 0.35 的目标
- [x] 新增 API 客户端 `generate_continuation()` 方法、自动重试、模型版本追踪
- [x] 新增 `predictor/llm.py`：`LlmCharPredictor` 和 `LlmTokenPredictor`
- [x] 新增测试：`test_online.py`（mock 单元测试）、`test_online_integration.py`（真实 API 测试）
- [x] CLI 新增 `--sub-mode` 选项，`bench` 命令支持在线模式对比

### v0.3.0 — 压缩率优化

- [ ] 预置中文字符频率表
  - 从大规模中文语料统计 top-3000 字符频率
  - 作为 order-0 模型的初始先验，改善短文本压缩
  - 硬编码到 `predictor/priors.py`，不增加文件体积
- [ ] 短语级编码
  - 识别高频短语（2-8 字符），作为单一符号编码
  - 需要修改逃逸编码机制以支持多字符符号
- [ ] order-5/order-6 实验
  - 评估更高阶上下文对长文本的收益
  - 注意内存占用和短文本退化

### v0.4.0 — 性能优化

- [ ] Rust 核心（PyO3）
  - 将 `arithmetic.py` 和 `bitstream.py` 用 Rust 重写
  - 通过 `maturin` + `pyo3` 构建 Python 扩展
  - 目标：10x 编码/解码速度提升
- [ ] 流式压缩/解压
  - 支持大文件分块处理（当前全文加载到内存）
  - 每块独立编码，支持随机访问
- [ ] 并行压缩
  - 多文件并行处理

### v0.5.0 — 工具链完善

- [ ] `pip install zippedtext` 直接从 PyPI 安装
- [ ] 命令行自动补全（click 原生支持）
- [ ] `zippedtext convert` — 批量压缩目录
- [ ] `zippedtext diff` — 比较两个 .ztxt 文件
- [ ] 压缩率可视化报告（每个字符的 bit 消耗热力图）

### v1.0.0 — 稳定版

- [ ] .ztxt 格式 v2（如有必要的破坏性改进）
- [ ] 完整的 API 文档
- [ ] 80%+ 测试覆盖率
- [ ] 跨平台 CI（GitHub Actions：Windows / macOS / Linux）
- [ ] 性能基准回归测试

### 远期方向

- 多模态支持：压缩 Markdown 中嵌入的结构化内容（表格、代码块）
- 自训练模型：根据用户历史文本训练个性化预测模型
- WebAssembly 版：浏览器端压缩/解压
- 与主流编辑器集成（VS Code 扩展：保存时自动压缩）

---

## 7. 常见问题

### Q: 为什么不用 BPE tokenizer？

当前使用字符级编码 + 逃逸机制。BPE tokenizer（如 DeepSeek 的）的优势在于子词切分可以减少 token 数量，但它引入了两个问题：(1) 需要在文件中存储 tokenizer 配置或依赖特定模型版本；(2) `encode → decode` 可能不完全可逆（UNK token）。字符级方案保证了完美的可逆性和零外部依赖。

### Q: 为什么短文本（< 100 字符）压缩率不好？

每个新字符首次出现需要编码 `ESCAPE + codepoint`（约 17-20 bits 对 CJK）。短文本中大部分字符只出现 1-2 次，逃逸开销占比高。长文本中字符复用率高，自适应模型学到的上下文模式更多，压缩率显著提升。预置中文频率表（v0.3.0 路线图）将改善此问题。

### Q: 能否压缩二进制文件？

不能。当前设计专为 Unicode 纯文本优化。二进制文件请使用 zstd 或 brotli。
