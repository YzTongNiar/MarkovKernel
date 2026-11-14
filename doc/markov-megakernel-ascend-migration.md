# MarkovMegaKernel（WeightsThatStay）昇腾 A3/A5 迁移文档

> 源论文：*WeightsThatStay: On-Chip Megakernels for Serial Heads in Speculative Decoding*（arXiv:2611.00506）
> 目标硬件：昇腾 910C（Atlas A3 代际，DAV_2201 双 Die 合封）与昇腾 950PR/950DT（A5 代际，DAV_3510，第三代达芬奇）
> 关联文档：[weights-that-stay-insight.md](weights-that-stay-insight.md)（论文洞察）、`../runs/arch-diagram/weights-that-stay-architecture.html`（实现架构图）

## 1. 迁移对象：三件需要搬的东西

论文在 H100/B200 上的收益来自三个可分离的机制，迁移时逐一映射：

| # | 机制 | 论文中的贡献 | 昇腾迁移难度 |
|---|---|---|---|
| M1 | 整 walk 单 kernel（消除步间启动边界） | 1.49× | 低：Ascend C 单 kernel 内循环即可 |
| M2 | 权重行分片驻留片上，跨步复用 | 再 2.53× | 中：片上容量账本需重算（见 §3） |
| M3 | 跨核分布式状态推进（prepare→score→select→commit）+ 全局选择同步 | 使 M1/M2 成立 | **高**：昇腾无 cooperative launch 官方等价物，跨 AICore 同步原语需要重新设计（见 §4.2） |

## 2. 硬件对照：H100/B200 → 昇腾 A3/A5

### 2.1 计算与存储层级映射

| GPU 概念 | 昇腾对应 | 说明 |
|---|---|---|
| SM / 持久 CTA | AI Core（AIC + 2×AIV 组） | 910B2：24 Cube / 48 Vector 核；950PR Server：32 Cube / 64 Vector [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=CubeCore%20%E6%A0%B8%E6%95%B0%7C24%7C28%7C32 "citation") |
| RF（256 KB/SM，驻留主仓） | 无直接对应；由 L1 + UB 承担驻留 | 昇腾寄存器堆不作为大容量驻留仓开放 |
| SMEM（228 KB/SM） | L1 Buffer 512 KB/核 + UB 192 KB（910B2）/ 248 KB（950，每 AIV 可用值） | L1 是 Cube 输入缓存，UB 是 Vector 工作区 [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=L1%7C512%20KB%7C512%20KB%7CCube%20%E8%BE%93%E5%85%A5%E7%BC%93%E5%AD%98 "citation") |
| TMEM（B200 专有） | L0C（128 KB → 950 翻倍至 256 KB） | 同为矩阵指令累加/输出驻留仓 |
| wgmma / mma.sync / tcgen05 | Cube 指令（L0A/L0B → L0C） | operand 位置约束同样决定 tile 映射 |
| TMA | MTE/DMA（GM↔L1/UB） | 部分驻留的流送通道 |
| L2（H100 50 MB / B200 更大） | L2 192 MB（910B2）/ 128 MB（950 Server）/ 112 MB（950 PCIE） | **差异最大的点，见 §3.2** |
| HBM 3.35 TB/s（H100） | 3.2 TB/s（910C）；4 TB/s（950DT）；1.6 TB/s（950PR） | 950DT 面向 Decode 配 144 GB/4 TB/s，950PR 面向 Prefill 配 128 GB/1.6 TB/s [昇腾950 NPU架构白皮书](kimi-cite://web?url=https%3A%2F%2Fpublic-download.obs.cn-east-2.myhuaweicloud.com%2Fascend%2F%25E6%2598%2587%25E8%2585%25BE950%2520NPU%25E6%259E%25B6%25E6%259E%2584%25E7%2599%25BD%25E7%259A%25AE%25E4%25B9%25A6.pdf&text=%E6%98%87%E8%85%BE950DT%E5%88%99%E8%81%9A%E7%84%A6%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A8%E9%87%8F%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%A4%8D%E6%9D%82%E6%8E%A8%E7%90%86%EF%BC%8C%E9%85%8D%E5%A4%87144GB%E3%80%814TB%2Fs%E9%AB%98%E9%80%9F%E7%89%87%E4%B8%8A%E5%86%85%E5%AD%98 "citation") |

### 2.2 代际与编程模型

- A2/A3（910B/910C）同属 DAV_2201，910C 为双 Die 合封，片内 128 GB HBM，任意两 NPU 双向互联 784 GB/s；A5（950PR/950DT）为 DAV_3510，SIMD/SIMT 混合编程，新增 SSBuffer（256 KB）作为 AIC↔AIV 核间消息通路 [微信公众平台](kimi-cite://web?url=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzIxOTQ0MTMxMw%3D%3D%26mid%3D2247485588%26idx%3D1%26sn%3D71c96abfe1be94ab602d8942a817b629%26chksm%3D96be62134ee3eb4985f6ef2eb03c0b42676efa41bd44c3c457df5b21e6107cfefb3f01d1b2db%26mpshare%3D1%26scene%3D23%26srcid%3D0817ENNhUCM7FzryC3ixqajx%26sharer_shareinfo%3D9ce555dc8bdd2e6485f352045dda27c6%26sharer_shareinfo_first%3D9ce555dc8bdd2e6485f352045dda27c6%23rd&text=%E7%89%87%E5%86%85%20128GB%20HBM%EF%BC%9B%E4%BB%BB%E6%84%8F%E4%B8%A4%20NPU%20%E5%8F%8C%E5%90%91%E4%BA%92%E8%81%94%20784%20GB%2Fs "citation") [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=SSBuffer%7C256%20KB%7CDAV_3510%20%E6%96%B0%E5%A2%9E%20AIC%E2%86%94AIV%20%E6%A0%B8%E9%97%B4%E6%B6%88%E6%81%AF%E9%80%9A%E8%B7%AF "citation")
- 容量典型值存在 SKU 裁剪：算子运行时必须用 `GetCoreNumAiv()` / `GetCoreMemSize` 取实际值，禁止硬编码 [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=%E7%AE%97%E5%AD%90%E8%BF%90%E8%A1%8C%E6%97%B6%E5%BF%85%E9%A1%BB%E7%94%A8%20GetCoreNumAiv()%20%E5%8F%96%E5%AE%9E%E9%99%85%E5%80%BC "citation")

## 3. 容量账本（迁移可行性的核心）

以论文主负载为基准：V=151,936，r=256 → bf16 投影 77.8 MB，行缩放 int8 后 38.9 MB。

### 3.1 片上驻留容量估算

| 平台 | 可驻留片上存储（聚合） | int8 38.9 MB 全驻留？ |
|---|---|---|
| H100（132 SM × (RF 256 KB + SMEM 228 KB) 理论 63.9 MB，论文实测工作区预留后可行） | — | 可行（论文方案） |
| A3 单 Die（24 核 × (L1 512 KB + L0C 128 KB) + 48 AIV × UB 192 KB ≈ 15.4 + 9.2 MB ≈ 24.6 MB） | ~24 MB/ Die | **不可行**，需部分驻留或 int8 + 双 Die 协同 |
| A5（32 核 × (L1 512 KB + L0C 256 KB) + 64 AIV × UB 248 KB ≈ 24.6 + 15.9 MB ≈ 40 MB） | ~40 MB | **勉强可行**（与执行工作区的争夺会很紧，参考论文 RF:SMEM 配比权衡） |

### 3.2 关键差异：昇腾的大 L2 改变了基线

H100 的 L2 只有 50 MB，装不下 77.8 MB 的 bf16 投影，所以参考实现每步都从 HBM 流送整张表——这是论文收益的大头来源。昇腾 910B2 的 L2 为 192 MB、950 Server 为 128 MB，**bf16 投影全量装入 L2 还有富余** [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=L2%7C192%20MB%7C128%20MB "citation")。这正是论文在 B200 上观察到的现象："B200's larger L2 also strengthens the reference path."

推论：**迁移后的收益结构会变化**——

1. M1（融合/消启动）收益保留，与硬件无关；
2. M2 的增量从"HBM 流量 → 0"缩小为"L2→L1/UB 重复搬运 → 0 + L2 访问延迟消除"，预期小于论文的 2.53×；
3. 因此第一步必须先做 **L2 驻留基线**的测量（Ascend 上几乎零成本：投影权重自然缓存进 L2），把"片上驻留 vs L2 驻留"的差值作为 M2 的真实上限。

### 3.4 各级片上缓存的逐层适配

把论文的"层次化 tile 映射"落到昇腾，每一级的角色和容量约束如下（容量为每核典型值，实际以 `GetCoreMemSize` 为准 [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=L1%7C512%20KB%7C512%20KB%7CCube%20%E8%BE%93%E5%85%A5%E7%BC%93%E5%AD%98 "citation")）：

| 层级 | A3（910C/Die） | A5（950） | 在迁移方案中的角色 |
|---|---|---|---|
| L0A/L0B | 64 KB ×2 | 64 KB ×2 | Cube 每步 GEMV 的 operand 暂存（latent + 当前 weight tile）；不承担跨步驻留，决定每步 staging 节奏 |
| L0C | 128 KB | **256 KB** | Cube 累加/输出；对应 B200 的 TMEM 角色，可驻留本分片的部分中间结果，但与输出工作区互相挤占 |
| L1 | 512 KB | 512 KB | **权重驻留主仓**（Cube 输入缓存，对应 H100 SMEM 的 wgmma 路径） |
| UB | 192 KB/AIV ×48 | 248 KB/AIV ×64 | Vector 侧：latent、选择 scratch；存权重则需每步 UB→L1→L0 倒手（对应论文 SMEM→RF 的 mma.sync 路径代价） |
| SSBuffer | 无 | **256 KB** | AIC↔AIV 组内消息通路：组内 token/score 交换可少走 GM；不能做全局归约 |
| L2 | 192 MB | 128 MB（Server） | 基线驻留层 + 全局同步区/gather 缓冲所在地 |

**分片算术（V=151,936，r=256，int8 行缩放）**：每行 256 B；38.9 MB 总表。

- **A3 单 Die（24 核）**：若只驻留 L1，每核分片 = 151,936/24 ≈ 6,331 行 ≈ 1.58 MB，远超 L1 的 512 KB；叠加 L0C 与组内 UB 后每核组可用约 1.0–1.2 MB，仍不够 → **部分驻留是默认形态**，驻留比例约 60–70%，其余经 MTE/DMA 每步流送（对应论文 248k 词表场景的处理）。
- **A5（32 核）**：每核分片 ≈ 4,748 行 ≈ 1.22 MB；L1 512 KB + L0C 256 KB + 组内 UB 2×248 KB ≈ 1.26 MB → **int8 全驻留刚好压线**，执行工作区（累加器、latent、选择 scratch、双缓冲）会与驻留分片正面争夺——与论文 §5.4 的 RF:SMEM 配比权衡同构，需要实测确定配比，必要时应答退回部分驻留。
- **大词表（248,320 行，int8 ≈ 63.5 MB）**：A3/A5 都不可能全驻留，直接走部分驻留路径；A3 每步流送量约为论文 H100 场景的 1.5×，L2 命中可显著缓冲。

### 3.5 精度路径多了一个选项

论文在 H100 上选 int8 而非 FP8 E4M3（钟形权重分布下 E4M3 把 75% 量化级浪费在只覆盖 12–17% 权重的区间）。A5 原生支持 HiF8（官方口径"保持 FP8 高效的同时精度非常接近 FP16"）和 INT8（865–1034 TOPS）[华为](kimi-cite://web?url=https%3A%2F%2Fwww.huawei.com%2Fcn%2Fnews%2F2025%2F9%2Fhc-xu-keynote&text=HiF8%EF%BC%8C%E5%9C%A8%E4%BF%9D%E6%8C%81FP8%E7%9A%84%E9%AB%98%E6%95%88%E7%9A%84%E5%90%8C%E6%97%B6%EF%BC%8C%E7%B2%BE%E5%BA%A6%E9%9D%9E%E5%B8%B8%E6%8E%A5%E8%BF%91FP16 "citation") [昇腾950 NPU架构白皮书](kimi-cite://web?url=https%3A%2F%2Fpublic-download.obs.cn-east-2.myhuaweicloud.com%2Fascend%2F%25E6%2598%2587%25E8%2585%25BE950%2520NPU%25E6%259E%259E%25E6%259E%2584%25E7%2599%25BD%25E7%259A%25AE%25E4%25B9%25A6.pdf&text=INT8(TOPS)%20865%2F756 "citation")。迁移时应三路对比：行缩放 int8 + 双平面 latent（论文方案）/ HiF8 单平面（可能免掉双平面技巧）/ bf16 L2 驻留（不量化，靠 L2 容量）。

## 4. 机制映射与替代件设计

### 4.1 持久内核（M1）

- Ascend C 单 kernel，`blockNum` = 实际 AICore 数（运行时查询，勿硬编码），核内 for 循环执行 γ 步；每核按词表行静态分片。
- 风险：vLLM Ascend 的 kernel 发射模型下，一个占满全部 AICore 且长时间存活的 kernel 与其他算子的并发/抢占关系需要在引擎侧保证排他（等价于论文 cooperative launch 的存活条件：任何时刻所有属主核都在场，否则跨核等待死锁）。

### 4.2 跨核全局选择（M3 的硬点）

论文用 64-bit key（分数保序位 + 反转 token id）原子 max + release 到达计数，并用一次对齐 16B 向量 load 轮询（PTX 未保证的硬件行为，可移植版本 +6.7% 延迟）。昇腾替代件：

- **首选方案（两阶段，安全）**：每核在 UB 做局部 argmax → 结果写入 GM 同步区 → 全核 barrier（GM flag + 内存屏障）→ 每核读出全部 winner 本地归约。比论文的多一次 GM 往返，但无未保证行为；
- **A5 加速器**：SSBuffer（256 KB，AIC↔AIV 核间消息通路）可用于组内 AIC/AIV 协调，减少 GM 往返次数 [稀土掘金](kimi-cite://web?url=https%3A%2F%2Fjuejin.cn%2Fpost%2F7685342914482225161&text=SSBuffer%7C256%20KB "citation")；全局归约仍走 GM/L2；
- **待验证项**：昇腾 GM 原子操作是否支持 64-bit max；若支持，可直接复刻 {maxKey, count} 单点方案。
- 采样路径（两级 inverse-CDF + Gumbel-max + Philox 键控）与硬件无关，可整体照搬，注意保持"选择即同步边界"与精确投机采样分布。

### 4.3 部分驻留流送

非驻留 tile 每步经 MTE/DMA 从 GM 流送到 L1，双缓冲流水掩盖延迟；驻留子集与 staging 空间争夺 L1/UB 的权衡与论文 §3.3 相同。A3 单 Die 片上 ~24 MB < 38.9 MB（int8），所以 **A3 上部分驻留是默认形态**；A5 上可把全驻留作为目标形态。

### 4.4 循环头（RNN/GRU）状态交换

分片隐状态 gather → GM 缓冲 + barrier；由于昇腾 L2 大，gather 缓冲大概率命中 L2，跨核交换代价可能低于论文在 H100 上的实测（论文中这是驻留后的新瓶颈，DeLS-Spec/Domino 每步 3 次跨 CTA 往返）。

## 5. vLLM Ascend 集成路径与验证分层

按论文消融的层次自底向上验证，每层独立量化（避免把驻留收益记到融合头上）：

1. **L0 基线 profile**：在 vllm-ascend 投机解码路径（EAGLE/MTP）上用 profiler 测草稿头投影的单步耗时与轮次占比（对照论文 Figure 3 方法），确认"小状态 × 大静态权重"的带宽瓶颈在昇腾上成立；profiler 运行遵守独占 lease 与时间窗标注纪律。
2. **L1 整 walk 单 kernel**：只做融合，权重不驻留 → 对应论文 1.49× 档；
3. **L2 驻留基线**：L1 基础上让投影自然驻留 L2（昇腾大 L2 零成本获得）→ 这是昇腾特有的中间档；
4. **L3 片上驻留 + 跨核选择**：完整方案 → 测量相对 L2 档的真实增量。
5. **正确性门槛**：int8/HiF8 路径的接受率对齐（论文方法：95% prompt-cluster bootstrap 置信区间包含 0）；Philox 采样在图回放/多次调用下分布正确。
6. **端到端**：vLLM Ascend serving 吞吐，扫 batch × walk 长度；预期增益低于论文的 12.3%（L2 基线更强），950PR（1.6 TB/s）上的带宽瓶颈更重、相对收益可能高于 950DT。

适用范围提醒：本方案针对"串行头 + 共享大投影"结构（DSpark/Domino/Markov 头、EAGLE/MTP 的共享 LM head 投影段）；MTP 中完整 transformer 层不在本文方法覆盖范围内。

## 6. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| 无 cooperative launch 等价物，跨核等待存在死锁前提 | 高 | 全核单 kernel + 引擎侧排他；先写最小同步原型验证 |
| GM 64-bit 原子 max 支持未知 | 中 | 两阶段归约兜底（§4.2） |
| 大 L2 摊薄 M2 收益 | 中 | L2 驻留基线先行，用数据决定是否值得做片上驻留 |
| 持久 kernel 与 serving 调度共存 | 中 | 限制在投机轮次的草稿阶段；与调度器对齐发射时机 |
| A3 双 Die 间同步代价未量化 | 中 | 首版限定单 Die；双 Die 版作为后续项 |
| SKU 容量裁剪 | 低 | 运行时查询容量，Tiling 参数化 |

## 7. 行动项

- [ ] MindIE Knowledge 知识库校验（当前会话 native entry 绑定失败，待可用后复核本文容量数字与同步原语结论）
- [ ] 在 vllm-ascend 投机解码路径 profile 草稿头投影占比（L0）
- [ ] 最小原型：Ascend C 持久 kernel + GM 两阶段全局 argmax 的可行性与死锁验证
- [ ] 容量实测：`GetCoreMemSize` 取目标 SKU 的 L1/UB/L0C 实际值，重算 §3.1 账本
- [ ] 精度三路对比实验：int8 双平面 vs HiF8 vs bf16-L2（投影误差 + 接受率）
- [ ] L1/L2/L3 分层性能验证，产出每层增量数据
