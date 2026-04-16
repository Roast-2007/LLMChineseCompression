# ZippedText 在线模式问题分析、LLM 深度集成路线图与后续改造计划

> 文件目的：系统整理当前 online mode 压缩率/性能/可复现性问题，解释为什么现在不得不存储 prediction cache，以及为什么这会让压缩率变差；同时给出一份面向“深度集成 LLM 到压缩算法所有部分”的新路线图。
>
> 面向对象：项目维护者 / 未来版本设计者 / 后续 AI Agent
>
> 最近状态更新：2026-04-16（v0.3.6）

---

## 0. 结论先行

当前 online mode 的核心问题，不是“模型不够强”这么简单，而是 **整个算法结构里 LLM 不能只被浅层地当成一个外部续写器/概率增强器使用**。

好消息是，这条路线已经在项目里发生实质转向：
- legacy char/token 已经冻结为兼容路径；
- structured online 已经成为主路径；
- v0.3.3 完成了 segment router / template codec / residual 基础架构 / v3 side info section 化；
- v0.3.4 进一步把 LLM 的作用从 `template_hints` 扩展到 **schema / slot hints**，并开始把 template slot 从普通字符串升级为 **typed slot codec**。

一句话总结当前阶段：

> **项目已经不再是“拿 LLM 给 PPM 打补丁”，而是开始让 LLM 参与结构化表示的构建；但真正的大收益还取决于 record family、global gain 和 hierarchical residual。**

---

## 1. 当前 online mode 的真实现状

### 1.1 现在 online mode 实际在做什么

当前 online mode 本质上分成两条路线：

#### A. legacy online（兼容）
- `char` / `token` 子模式仍保留；
- 依赖 prediction cache；
- 用于兼容旧思路、对比效果和回归测试；
- **不再作为主架构继续扩展**。

#### B. structured online（主路径）
- LLM 在压缩期做一次 structured analysis；
- analysis 结果会进入：
  - 字符先验融合
  - phrase / term dictionary 排序
  - segment routing
  - template detection
  - template payload 设计
- `.ztxt` v3 保存 analysis / dictionary / template / segment / stats sections；
- 解压完全依赖文件内 side info 和本地确定性 coder，不访问远端 API。

### 1.2 v0.3.4 之后 structured online 已经新增的能力

在 v0.3.4，structured online 的 analysis manifest 不再只有 `template_hints`，而是开始支持：
- `document_family`
- `block_families`
- `field_schemas`
- `slot_hints`
- `enum_candidates`

同时，template slot 开始支持 typed slot payload：
- `identifier`
- `version`
- `path_or_url`
- `enum`
- `number_with_unit`

这意味着当前系统已经迈出关键一步：

> **LLM 不只是在提示“这行像模板”，而是在提示“这个字段/槽位是什么类型，应该如何更短地表示”。**

---

## 2. 当前已知问题总表（更新版）

下面按“压缩率、结构表示、全局收益、工程化验证”四个层面归纳当前最关键的问题。

### 2.1 已经不是主要问题的点
这些点在当前主路径里已经不是最核心瓶颈：
- “side info 太重所以 structured online 完全没意义”
- “必须继续围绕 legacy char/token cache 深调”
- “structured online 只能存一些弱提示，无法真正参与编码”

原因：
- v3 section 化已经显著降低了 structured side info 的无序膨胀；
- analysis priors 已经真正并入 structured coder；
- template / residual / gain estimator 已经形成基础闭环；
- v0.3.4 已开始把 slot 从 string 升级为 typed payload。

### 2.2 现在真正限制压缩率上限的 5 个点

1. **模板仍然太偏单行**  
   当前 template 复用半径主要停留在 line-level key-value / list / table row。很多真实文档的重复，不是单行重复，而是“多行记录结构重复”“段落骨架重复”“API 条目重复”“术语说明块重复”。

2. **typed slot 只是第一步，slot value 仍未被充分解释**  
   v0.3.4 已经让 `version` / `path_or_url` / `enum` / `number_with_unit` / `identifier` 开始可压缩，但 typed residual 仍不够强，slot 命中后仍可能保留较多字面成本。

3. **收益估计仍然偏 segment-local**  
   当前 router 已有 family amortization 雏形，但仍主要以当前 segment 的净收益做决策。很多 codec 的价值必须在“整份文档”或“一个 block family”上摊薄后才体现出来。

4. **LLM 还没有给出真正的 record/document-level 可逆结构表示**  
   analysis 已经从 hint 层向 schema 层推进，但当前还没有完成 record family / paragraph skeleton / document-level family clustering。

5. **跨段、跨块、跨章节复用还不够强**  
   真实技术文档、API 文档、配置说明、双语术语表的重复，经常表现为“同一字段族”“同一条目骨架”“同一解释句式反复出现”。当前复用能力还不足以系统性吃掉这种长程重复。

---

## 3. 当前已完成标注（2026-04-10 / v0.3.4 typed slot 落地）

### 已完成
- [x] **冻结 legacy online mode**
  - `online-legacy-char`
  - `online-legacy-token`
- [x] **引入 structured online 主路径**
  - online 默认子模式已切到 `structured`
- [x] **引入 segment router**
  - 已新增 `src/zippedtext/segment.py`
  - 已新增 `src/zippedtext/router.py`
- [x] **引入 LLM 辅助 term / phrase dictionary 初版**
  - 已新增 `src/zippedtext/term_dictionary.py`
  - `ApiClient.analyze_text()` 已接入 structured online
- [x] **引入 `.ztxt` v3 初版**
  - 支持 typed sections
  - 当前已实现 analysis / phrase_table / templates / segments / stats section
- [x] **structured online 解压不依赖远端 API**
- [x] **补齐 structured online 测试**
  - `test_analysis_manifest.py`
  - `test_router.py`
  - `test_online_structured.py`
  - `test_format_v3.py`
- [x] **template codec 第一轮闭环**
  - 当前支持 key-value / list prefix / table row
- [x] **residual architecture 第一轮闭环**
  - template payload 采用 `template_id + slot_values + residual`
  - residual 复用现有 literal / phrase coder
- [x] **online gain estimator 第一轮落地**
  - 支持 literal / phrase / template 净收益选路
  - `info` / `bench` 已可展示 side info 拆解、route 分布、template hit、residual bytes
- [x] **stronger structured side-info compression**
  - segment records / stats 已改紧凑二进制
  - v3 section 支持 raw/zstd codec 与 flags
- [x] **v0.3.3 第二阶段增强**
  - analysis priors 真正并入 structured coder
  - `list` / `table` / `config` 优先行级切分
  - template detection 改为 hint-aware scoring
  - template catalog 会过滤明显 one-off 模板
  - phrase ranking 会结合 `phrase_dictionary` / `top_bigrams` / `char_frequencies`
  - fallback reason 更细，可区分 `literal best` / `side-info cost too high` / `template no catalog reuse`
  - 已补 structured online API smoke test
- [x] **v0.3.4 typed slot / schema hint 增强**
  - `online_manifest.py` 新增 `document_family` / `block_families` / `field_schemas` / `slot_hints` / `enum_candidates`
  - `api_client.py` analysis prompt 已开始请求 schema / slot hint
  - `template_codec.py` 已支持 typed slot payload：
    - `identifier`
    - `version`
    - `path_or_url`
    - `enum`
    - `number_with_unit`
  - `router.py` / stats 已开始记录 typed slot / typed template / template family 统计
  - `zippedtext info` / `bench` 现在可显示 typed slot 观测值
  - `bench` headline 已改为走 public structured path，并保留 raw structured diagnostic

### 仍未完成
- [ ] benchmark matrix 完整报表
- [ ] 更完整模板体系（跨行列表、文档段模板、更多配置模板、API 文档专用模板）
- [ ] record grouping / multi-line record template
- [ ] 更强的 family-level / global gain optimization
- [ ] entity / alias / bilingual term system
- [ ] hierarchical residual
- [ ] MoE probability layer
- [ ] 本地确定性模型

### 当前实现边界
当前 v0.3.4 已经证明 roadmap 主线正确：
- 瓶颈已经不再是“side info 太重”；
- 当前主要问题转成 **模板命中半径、slot value 编码、family 级收益摊销不足**；
- 继续深挖 typed slot / record family / global gain，比重新回头调 legacy char/token cache 更有价值。

---

## 4. 基于现有项目状态，后续 LLM 深度集成最值得投入的方向

### 方向 A：从 schema hints 升级到 document family / record family induction

#### 核心想法
让 LLM 不只返回“这里像 key-value / list / table”，还要稳定返回：
- 文档家族（API 文档 / 配置说明 / 教程 / FAQ / release notes / 双语术语表）
- block family
- 记录结构
- 字段集合与顺序规律
- 条目族之间的共用骨架

#### 为什么这一步很重要
因为当前 `template_codec.py` 的复用单位还太小；而真实文档的高重复结构常常是“多行记录”或“多段条目”。

#### 对现有代码的含义
- `api_client.py`：analysis 继续从轻量 hints 升级到 block family / schema family 候选
- `online_manifest.py`：继续扩展 `document_family` / `block_family` / `field_schema`
- `segment.py`：从“切段”升级到“切块 + 归类到 family”
- `router.py`：不只决定 route，还要决定 segment 属于哪个 family

#### 预期收益来源
- 同一家族 segment 共用一套模板 catalog / dictionary / field codec
- side info 在整文档尺度摊销
- 提高 template 命中率和 catalog 复用率

---

### 方向 B：在 typed slot 之上补齐 typed residual

#### 核心想法
v0.3.4 已经让 slot value 开始按类型表示，但当前 residual 仍主要是字面 fallback。下一步必须让 residual 也更懂类型：
- enum residual
- version residual
- path/url residual
- number-with-unit residual
- identifier residual

#### 为什么这一步最可能直接继续拉高压缩率
因为很多 structured 文本的问题已经不再是“命不中模板”，而是“命中了模板和 slot type，但剩余字面成本仍偏大”。

#### 对现有代码的含义
- `template_codec.py`：typed slot 解码器要输出更细的 residual 解释位点
- `residual.py`：从单层 residual 升级为 typed residual
- `router.py` / `gain_estimator.py`：收益估计要把 typed residual 一起算进去

---

### 方向 C：从 line template 升级到 multi-line record template / paragraph skeleton

#### 核心想法
很多真实语料的重复结构不是单行，而是类似：
- API 参数条目块
- 配置项说明块
- 标题 + 一到两行解释
- 中英术语 + 注释 + 示例
- release note 条目
- FAQ 项

应逐步升级到：

```text
record = record_template_id + typed_slots + record_residual
```

而不是每行单独 route。

#### 对现有代码的含义
- `segment.py`：需要支持 record grouping，而不是只 line split
- `template_codec.py`：新增 record template family
- `router.py`：先决定“按行 route”还是“按 record route”
- `gain_estimator.py`：支持 family-level / record-level 收益估计

---

### 方向 D：从 segment-local gain 变成 global optimization

#### 核心想法
当前 `gain_estimator.py` 主要在回答：**这一段现在值不值得走 template / phrase**。

但未来更重要的问题是：

> **如果我为这类 block 建一套 family template / dictionary / slot codec，整份文档总收益是否为正？**

也就是说，收益估计器需要从：
- single segment decision

升级到：
- document-level family decision
- batch amortization decision
- side-info budget allocation

#### 为什么这一步很关键
很多高级 codec 前几次命中可能不赚钱，但只要文档里有 20~50 个同类块，整体就会很赚。若没有全局优化，系统会长期偏向“保守但浅层”的 route。

---

### 方向 E：把 term dictionary 扩展成 entity / field / alias system

#### 核心想法
当前 `term_dictionary.py` 还是偏 phrase table。下一步应让 LLM 帮助构建：
- 字段名簇
- 参数名簇
- 实体简称 / 全称 / 别名映射
- 中英术语对照表
- 同文档里的重复说明短句簇

#### 预期收益来源
- 降低重复术语的原样存储成本
- 吃掉中英混合技术文档中的重复 naming 成本
- 为 typed slot codec 提供更稳定的字典基础

---

### 方向 F：给不同文档族做专用 LLM-assisted codec

#### 核心想法
当前 structured online 已经证明“按文本类型吃结构收益”是对的。下一步不应继续只做一个泛型模板系统，而应逐步拆出专用 codec：
- `api_doc_codec`
- `config_doc_codec`
- `bilingual_term_codec`
- `release_note_codec`
- `faq_codec`

#### 为什么会显著提高压缩率
因为不同文档族的重复规律完全不同：
- API 文档：字段、类型、路径、示例值
- 配置文档：key、default、description、enum
- 双语术语表：term pair + 注释
- FAQ：question / answer 模板

---

## 5. 一个更明确的判断：真正的大压缩率提升，最可能来自哪里

如果只问一句：

> **在当前项目基础上，下一阶段最可能让压缩率出现明显跃升的点是什么？**

我的判断是：

1. **multi-line / record template family**
2. **document-level family clustering + global gain optimization**
3. **typed residual**
4. **entity / alias / bilingual term system**
5. **family-specific codec**

也就是说：

> **真正的大收益不会来自“LLM 再多猜几个字符”，而会来自“LLM 把整份文档映射成更短、更稳定、更可复用的结构表示”。**

---

## 6. 推荐的下一步优先级（面向现有项目，不空谈）

### 第一优先级：先做这 4 件事
1. 在 `segment.py` 上增加 **record grouping**。  
2. 在 `template_codec.py` 上扩展 **multi-line / record template family**。  
3. 在 `router.py` + `gain_estimator.py` 里继续深化 **family-level / global gain optimization**。  
4. 在 `residual.py` 中引入 **typed residual**，不要让 slot 命中后仍大量回退字面 residual。  

### 第二优先级：再做这 3 件事
5. 扩展 `term_dictionary.py` 为 **entity / alias / bilingual term table**。  
6. 建一套 **benchmark matrix**，专门衡量：API 文档、配置行、双语术语表、release notes、FAQ。  
7. 继续增强 `info` / `bench` 的 family-level observability。  

### 第三优先级：之后再做
8. family-specific codec（API / config / term / FAQ）  
9. 本地小模型或更深的 MoE 概率层  
10. 再考虑 v4 文件格式升级  

---

## 7. 不建议优先投入的方向

以下方向不是没价值，而是**不是当前最可能带来“大幅压缩率提升”的首要点**：

- 继续围绕 legacy char/token cache 做微调
- 只靠 prompt engineering 继续挖 `template_hints`
- 在还没有 typed residual / family routing 前就急着做复杂 MoE
- 过早把解码重新绑回远端 API
- 只增加更多 template kind 名称，但不解决 record family 与 slot/residual 成本

一句话说：

> **真正值得优先做的，不是“让 LLM 再多预测一点”，而是“让 LLM 帮压缩器找到更短的可逆表示”。**

---

## 8. 新版本规划（按当前真实状态重写）

### v0.3.4 — schema-seeded typed slot structured online ✅
- schema / slot hints 落地
- typed slot codec 初版
- info / bench typed observability
- CLI / fixture / structured regression 补强

### v0.3.5 — code quality + robustness hardening ✅
- `_merge_priors()` 浮点精度阈值修复（`> 1e-15` 替代 `> 0`）
- `_match_key_value()` 嵌套括号安全处理
- structured 路径前 `_should_skip_structured_api()` 启发式预检
- `probs_to_cdf()` redistribution 极端情况兜底
- `_looks_like_config()` 行级模式检测修复
- `router.py` `detect_template()` 预扫描缓存（消除重复调用）
- `_count_all_phrase_occurrences()` 单次全文扫描优化
- 内部函数 `_get_priors` / `_structured_online_compress` 重命名为公共 API
- `Header.version` 默认值与 `VERSION` 常量同步

### v0.3.6 — record family 扩展期
- record grouping
- multi-line / paragraph template family
- typed residual 第一轮
- benchmark matrix 初版

### v0.3.6+ — family/global optimization
- document-level family clustering
- global gain optimization
- entity / alias / bilingual term table
- 更完整 family-specific codec

### v0.3.5 — code quality + robustness hardening ✅
- `_merge_priors()` 浮点精度阈值修复（`> 1e-15` 替代 `> 0`）
- `_match_key_value()` 嵌套括号安全处理
- structured 路径前 `_should_skip_structured_api()` 启发式预检
- `probs_to_cdf()` redistribution 极端情况兜底
- `_looks_like_config()` 行级模式检测修复
- `router.py` `detect_template()` 预扫描缓存（消除重复调用）
- `_count_all_phrase_occurrences()` 单次全文扫描优化
- 内部函数 `_get_priors` / `_structured_online_compress` 重命名为公共 API
- `Header.version` 默认值与 `VERSION` 常量同步

### v0.3.6 — record family expansion ✅
- `RecordGroup` 数据结构：连续同类型结构化段落分组
- `group_record_groups()` 函数：利用 `block_families` 标注 family
- `record` 模板 kind（ID: 3）支持多行记录模板
- `TemplateCatalog` 序列化支持 JSON-encoded per-line skeleton
- `_render_record_template()` / `_match_record_template()` 多行渲染与检测
- typed residual（`RESIDUAL_TYPED`）：enum / version / path_or_url / number_with_unit / identifier
- `router.py` 接受 `record_groups` 参数 + `_evaluate_record_group_route()`
- benchmark matrix：6 类结构化样本 + `bench.py` runner
- 154 个测试全部通过

### 更长期
- 本地确定性模型
- MoE probability layer
- 视需要考虑 v4 文件格式

---

## 9. 当前明确判断

1. 当前 online mode 主线已经不是 legacy char/token，而是 structured online。
2. v0.3.4 之后，最关键的收益瓶颈已经是 **typed slot 之外的复用半径和 family 级收益摊销**。
3. 继续围绕远端 API 解码做文章没有意义；API-free structured decompression 必须保持。
4. 真正的突破不会来自再调 `CHUNK_CHARS` 或 `boost factor`，而会来自：
   - record family
   - global gain optimization
   - entity / alias / bilingual term system
   - hierarchical residual
   - local deterministic modeling
5. 如果目标是“深度集成 LLM 到压缩算法中”，那么今后必须继续明确转向：

> **LLM 不只是预测器，而是压缩系统里的结构建模器、模板生成器、字典构建器、模式路由器和残差解释器。**

---

## 10. 推荐的下一步执行顺序（非常具体）

### 立刻做（v0.3.5 已完成代码质量修复，接下来进入结构扩展）
1. 在 `segment.py` 上增加 record grouping
2. 在 `template_codec.py` 上扩展 multi-line record template
3. 在 `router.py` / `gain_estimator.py` 中做 family-level / global amortization
4. 在 `residual.py` 中把 typed residual 做成真正可分层的结构
5. 建一套结构化 benchmark matrix（API/config/release-note/FAQ/term list）

### 接着做
6. 扩展 `term_dictionary.py` 为 entity / alias / bilingual term table
7. 引入 family-specific codec skeleton
8. 进一步增强 `info` / `bench` 的 family-level observability

### 再做
9. 研究本地确定性模型接入
10. 再考虑更深的 MoE 与文件格式升级

---

## 11. 最终目标应该是什么

真正的终局不是“让远端 LLM 多预测几个字符”，而是：
- 编码期可以充分使用 LLM 做结构建模
- 最终写入文件的 side info 仍然紧凑且可逆
- 解压完全不依赖远端 API
- structured path 在多个主要结构化语料族上稳定优于 offline

---

## 12. 附：一句话版路线图

> 先把 LLM 提供的 hints 升级为 schema / slot / family 级可逆结构，再把 line template 升级为 record family 与 global gain；真正的大压缩率提升将来自更短的结构表示，而不是更长的预测文本。
