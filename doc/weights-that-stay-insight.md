# WeightsThatStay：投机解码串行头部的片上 MegaKernel —— 论文洞察

> 论文：*WeightsThatStay: On-Chip Megakernels for Serial Heads in Speculative Decoding*
> 作者：Xinwei Qiang、Ziyi Xu、Zaifeng Pan、Chang Chen、Yufei Ding（UCSD / RadixArk）
> arXiv:2611.00506v1 [cs.DC]，2026-10-31。原文：[markov megakernel](markov megakernel.pdf)

## 一句话总结

并行投机采样的串行修正头（Markov / RNN / GRU）算力极小，却因每一步都重新搬运同一张词表级投影矩阵而成为延迟热点；WeightsThatStay 用一个持久化 megakernel 把投影权重分片**钉在片上**走完整个串行 walk，步间只交换紧凑的 token/状态，头部级加速最高 11.6×，端到端 SGLang 吞吐最高提升 12.3%。

## 1. 要解决的问题

- **背景结构**：DSpark、Domino 等"并行草稿 + 串行修正头"方案中，主干一次前向算出所有草稿位置的 base logits 和特征，随后由轻量串行头逐步修正：每步 `ℓk = bk + W·uk`，其中 W ∈ R^{V×r}（V > 150k 词表行，r=256）在整个 walk 中**不变**，跨步依赖只由很小的 token/循环状态承载。
- **痛点**：r=256 的 bf16 投影在 V=151,936 时占 77.8 MB，V=248,320 时占 127.1 MB，均超过 H100 的 50 MB L2。逐步执行 = 每步把这张大表重新流经整个存储层级。算术强度约 2B/p ops/byte，小 batch 下纯带宽瓶颈。实测在 SGLang 中该头占 Qwen3-4B 投机轮次延迟的**最高 12%**，单步词表投影 44.4 µs。
- **根因**（论文的定性判断，也是全文最有迁移价值的观点）：**执行边界与可复用数据的生命周期不匹配**——W 的复用周期是整个 walk，而常规算子边界把寄存器/SMEM 中数据的软件生命周期限制在单次 kernel 启动内。

## 2. 核心方法

### 2.1 Walk-resident 执行

把"整 walk 融合成单 kernel"与"权重驻留"两件事显式分开：消融证明整 walk 融合只带来 1.49×（省启动/中间流量），权重驻留再贡献 2.53×——**单发射不等于免搬运**。做法是 cooperative launch 一个与 SM 一一对应的持久 CTA 网格，按词表行把 W 切成固定分片，每个 CTA 整个 walk 只给自己的行打分。

### 2.2 层次化 tile 映射设计空间

- **SM 间**：按词表行静态划分所有权，walk 期间不变；
- **SM 内**：分片在 RF / SMEM / TMEM（B200 专有）之间分配，并由矩阵指令的 operand 位置约束决定（mma.sync 用 RF；wgmma 读 SMEM；tcgen05 用 TMEM）。映射同时决定驻留容量和每步的 operand 装载代价。

### 2.3 分布式状态推进协议

每步四阶段：prepare → score → select → commit。跨 CTA 选词用 64-bit key（分数保序位 + 反转的 token id 破平局）做原子 max，再用 release 序的到达计数；H100/B200 实现进一步用一次对齐 16B 向量 load 轮询 {key, count} 对，省一次 L2 往返——但这是**PTX 未保证的硬件行为**，可移植版本慢 6.7%。采样路径用两级 inverse-CDF + Gumbel-max，Philox 计数器按 (request, 行, 步, graph replay) 键控保证 CUDA Graph 重放时采样仍然新鲜，且保持精确投机采样分布。

### 2.4 容量不够时：表示压缩 + 部分驻留

- **H100 int8**：bf16 全量驻留不可能（77.8 MB > 片上 65.4 MB 聚合容量）。实测权重行近似钟形、无重尾，行缩放 int8（0.58% Frobenius 误差）显著优于 E4M3（2.64%，75% 量化级落在只覆盖 12–17% 权重的区间）。latent 用双 int8 平面编码（高平面 + 1/256 尺度的残差平面），保持 int8 tensor core 全程不反量化。
- **部分驻留**：装不下时固定一部分 tile 驻留，其余每步由 TMA 流送（H100 大词表场景每步固定 ~41 MB），驻留比例与 staging 缓冲相互争夺片上空间。

## 3. 关键结果

| 维度 | 结果 |
|---|---|
| 头部级（H100, Markov, γ=16, B=1） | 44.2 µs/步（SGLang）→ 2.69 µs/步，**最高 11.6×** |
| 消融：整 walk 融合 / 权重驻留 | 1.49× / 再 2.53× |
| 循环头（DSpark RNN / DeLS-Spec / Domino） | 8.9× / 8.2× / 8.5× |
| 端到端 SGLang 吞吐 | Qwen3-4B **+12.3%**，Qwen3-8B +8.6%（block-16 Markov） |
| 大词表（248k，部分驻留） | B=1 时 4.4×，随 batch 升到 ~5.8× 后回落 |
| B200（bf16 + TMEM） | 152k 全驻留；248k 约 2/3 驻留 |
| 数值影响 | int8 后接受率变化 95% 置信区间包含 0，无可测影响 |

收益随并发和目标模型变大而**递减**（head 占比被摊薄）；论文指出 TP 场景可能反转——目标计算被切分而头在每个 rank 上复制，占比反而升高。循环头驻留后，**跨 CTA 状态交换成为新的主要开销**（DeLS-Spec 与 Domino 每步 3 次跨 CTA 往返，延迟几乎一致）。

## 4. 局限与诚实标注

- cooperative launch 要求所有参与 SM 的 CTA 全程存活——资源独占性强，与并发 serving/混部天然紧张；
- 依赖未在 PTX 中承诺的 16B 向量 load 原子快照行为，可移植版本付出 6.7% 延迟；
- int8 适用性依赖权重分布形态（钟形无重尾），换了头需要重新验证；
- 只优化草稿头；服务侧集成仅验证了 SGLang，未覆盖 vLLM；
- 高并发下收益收窄至个位数百分比。

## 5. 对 vLLM / vLLM Ascend 的启发

1. **问题形态在 vLLM 中同样存在**：EAGLE/MTP 类草稿头都是"小状态 + 共享大投影"的串行 walk（DeepSeek/Qwen 的 MTP 头复用词表级 LM head 投影）。在 vLLM Ascend 上，NPU 侧 LM head / MTP 投影在大词表模型上同样受带宽支配，是值得优先 profile 的目标。
2. **方法可移植，原语需重写**：Ascend AI Core 的片上层次（L1/L0A/L0B/UB）对应 RF/SMEM/TMEM 的角色划分；核心思想——按词表行静态分片到核、权重全程驻留、步间只传 token/隐状态——可以直接映射为 AICore 持久 kernel。需要替代件：跨核全局选择（原子 max + 到达计数 → Ascend 的核间同步/原子原语）、部分驻留的流送路径（→ Ascend DMA/流水机制）、以及双平面 int8 GEMV（→ Ascend int8 cube 指令）。
3. **先把 fusion 与 residency 的收益分开量化**：论文消融（1.49× vs 2.53×）说明"融成一个大 kernel"和"权重不重复搬运"是两个独立的收益来源。在 Ascend 上做 drafter 头优化时应同样拆开测，避免把驻留的功劳记到融合头上。
4. **采样必须留在精确分布上**：他们的两级 inverse-CDF + Gumbel-max + Philox 键控方案，是把 GPU 内选择嵌入驻留 walk 同时不破坏投机采样正确性的完整参照——任何 NPU 侧重写都需要同等的"选择即同步边界"设计。
5. **更大的图景**：与 MPK（OSDI'26）、Event Tensor 同属 megakernel 潮流，但本文的增量是"权重驻留超越算子融合"的量化证据。对 vLLM Ascend 而言，decode 阶段小 batch 的任何"小状态 × 大静态权重"结构（不止投机头——例如低秩 adapter、gate 网络、KV 无关的固定投影）都可套用同一诊断：复用周期 > 算子生命周期 ⇒ 考虑持久化 + 驻留。

## 6. 可跟进的行动项

- [ ] 在 vLLM Ascend 投机解码路径上 profile 草稿头投影的单步带宽占用与轮次占比（对照论文 Figure 3 方法）；
- [ ] 调研 Ascend 上持久 kernel + 核间同步的可行原语（对照 cooperative launch 的存活/死锁条件）；
- [ ] 复现行缩放 int8 vs FP8 的投影误差对比，确认目标 drafter 权重分布是否满足钟形假设；
- [ ] 关注 TP 场景下"头被复制、目标被切分"的占比反转论断——多卡 Ascend serving 可能是该优化收益更大的场景。
