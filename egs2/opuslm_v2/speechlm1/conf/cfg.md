
- speechlm/model/speechlm/lm/parallel.py

## CFG Process
下面按这段代码实际怎么跑来解释它的 CFG（Classifier-Free Guidance）工作流程（重点在 inference_segment() 和 _prepare_cfg_cache() 这一套）。

⸻

1) CFG 在哪里开启？——inference_segment(config)

CFG 由 config.get("cfg", 1) 控制：

cfg = config.get("cfg", 1)
if cfg > 1:
    cache, cfg_logits = self._prepare_cfg_cache(cache)

	•	cfg <= 1：不启用 CFG，正常按条件（prompt / 上下文）采样。
	•	cfg > 1：启用 CFG，要同时得到条件预测和无条件预测，并按比例混合 logits。

这份实现把“无条件分支”的 logits 变量名叫 cfg_logits，容易误读；它其实就是 unconditional logits。

⸻

2) 无条件分支怎么构造？——_prepare_cfg_cache(cache)

核心目的：让“无条件分支”拥有与“条件分支”同样的 KV cache 长度（也就是同样的前缀长度），这样后续一步步生成时，两条分支能用同一个 step 逻辑并行计算。

length = cache.get_seq_length()
batch_size = cache.layers[0].keys.shape[0]

zeros = torch.zeros((batch_size, length, self.num_stream)).long().to(device)
cfg_logits, cfg_cache = self._step(input_ids=zeros)

这里的关键点：
	•	zeros 的 shape 是 [B, prefix_len, num_stream]
	•	全 0 token 在这个工程里等价于“空条件 / 无提示”（unconditional）。
（严格来说它是“全部 padding/空 token”的前缀，模型学到的就是一种无条件先验。）

然后把两套 cache 沿 batch 维拼起来：

key = torch.cat([cond_keys, uncond_keys], dim=0)
value = torch.cat([cond_values, uncond_values], dim=0)

得到 combined_cache，其 batch 维变成 2B，排列顺序是：
	1.	前 B：原来的 conditional cache
	2.	后 B：新算的 unconditional cache

同时 _prepare_cfg_cache 还返回了 cfg_logits（无条件 prefill 的 logits），用于某些“已经预先算好一步 logits”的场景对齐。

⸻

3) 生成循环里如何并行跑两条分支？——“把 batch 复制成 2 倍”

进入每个 step：

if cfg > 1:
    prev_token = prev_token.tile(2, 1, 1)
...
logits, cache = self._step(input_ids=prev_token, past_key_values=cache, mask=mask)

因为 combined_cache 的 batch 是 2B，所以 prev_token 也要复制成 2B 才能一一对应：
	•	前 B 个 token + 前 B 个 cache → 条件分支
	•	后 B 个 token + 后 B 个 cache → 无条件分支

然后把输出 logits 拆开：

logits, cfg_logits = logits.chunk(2)

此时：
	•	logits：conditional logits
	•	cfg_logits：unconditional logits

⸻

4) CFG 的 logits 混合公式是什么？

代码用的是：

logits = logits * cfg + cfg_logits * (1 - cfg)

把它整理一下：

\text{mixed} = cfg \cdot \text{cond} + (1-cfg)\cdot \text{uncond}
= \text{uncond} + cfg \cdot (\text{cond} - \text{uncond})

这正是经典 CFG 形式：

\text{uncond} + s(\text{cond}-\text{uncond})

所以这里的 cfg 就是常说的 guidance scale（引导强度）：
	•	cfg = 1：mixed = cond（完全等于条件分支，不加引导）
	•	cfg > 1：放大 (cond - uncond)，更“听 prompt”
	•	cfg 越大：越贴合条件，但通常多样性下降、也更容易出“过饱和/奇怪伪影”（在图像扩散里尤其明显；这里是离散 token 生成也有类似权衡）

混合完再做一次 mask：

logits.masked_fill_(mask, float("-inf"))

确保只在当前 modality 允许的 token 集合里采样。

⸻

5) precomputed_logits 时 CFG 怎么对齐？

这段模型有一种模式：某些情况下外面已经算好了“下一步的 logits”（比如 continuation 模式把最后一步 logits 先算出来并 mask 过），这时 inference_segment 会收到 precomputed_logits。

CFG 开启后，它会把无条件 prefill 的 logits 也拼上去，保证 batch 维仍然是 2B：

cache, cfg_logits = self._prepare_cfg_cache(cache)
precomputed_logits = torch.cat([precomputed_logits, cfg_logits[:, -1:, :, :]], dim=0)

注意这里取 cfg_logits[:, -1:, :, :]：只拿无条件 prefill 的最后一个位置，因为 precomputed_logits 通常只有最后一步 [B, 1, S, V]。

⸻

6) 多假设（num_hypo > 1）与 CFG 一起时怎么处理？

代码先处理多假设：
	•	把 prev_token tile 成 num_hypo
	•	cache.batch_select_indices(indices) 让 cache 的 batch 也扩成 num_hypo

然后如果再启用 CFG，会再扩成 2 倍（条件/无条件），最终 batch 维是：

B_\text{final} = 2 \times num\_hypo

并行一次 forward，就能拿到每个 hypothesis 的 cond/uncond logits，再做 CFG 混合。

⸻

7) 这套实现的 CFG “语义”总结（对应你前面那段概念解释）
	•	Conditional 分支：用真实的上下文 cache（prompt + 历史 token）去预测下一 token。
	•	Unconditional 分支：用全 0 token 构造的“空上下文 cache”去预测下一 token（更像模型先验）。
	•	CFG：用 uncond + cfg*(cond-uncond) 把生成朝条件方向“推”过去。

## CFG graph

下面给你一张“带张量形状的 CFG 工作流程图”（按你这份代码的真实数据流来画），用 ASCII 流程图表示：

记号：
	•	B = batch size（通常 1）
	•	T = 当前序列长度（prefill 长度 / 每步是 1）
	•	S = num_stream
	•	H = hidden size
	•	V = vocab size
	•	Nh = num_hypo（多假设条数）
	•	cache 的 batch 维会跟着一起扩

⸻

0) Prefill（在 inference() 里先把上下文跑进 cache）

kwargs["seqs"] = input_ids: [B, T0, S]
        |
        v
_embed(input_ids, kwargs)
  - embed_tokens + sum(streams) + conti adaptor ...
inputs_embeds: [B, T0, H]
        |
        v
_step(input_embeds=..., past_key_values=None)
  -> transformer(use_cache=True)
  -> hidden: [B, T0, H] -> unsqueeze -> [B, T0, 1, H]
  -> + stream_emb -> [B, T0, S, H]
  -> lm_head -> logits: [B, T0, S, V]
  -> cache: DynamicCache(batch=B, seq_len=T0)

到这里为止：你有了“条件分支”的 cache（因为它来自真实上下文）。

⸻

1) 进入 inference_segment()：先处理多假设 + CFG 预处理

1.1 多假设（可选：num_hypo > 1）

prev_token: [B, 1, S]
cache: batch=B
        |
        | if Nh > 1:
        v
prev_token = tile(Nh) -> [Nh*B, 1, S]
cache.batch_select_indices -> cache batch = Nh*B
(precomputed_logits 若有也 tile) -> [Nh*B, 1, S, V]

此时 batch 变成 B' = Nh*B。

⸻

1.2 CFG 预处理（可选：cfg > 1）—— _prepare_cfg_cache(cache)

目标：构造一套 unconditional cache，并和 conditional cache 拼成一个 2B' 的大 cache。

已有 conditional cache: DynamicCache(batch=B', seq_len=L)
        |
        v
zeros = 0-token 前缀: [B', L, S]
        |
        v
_step(input_ids=zeros, past_key_values=None)
  -> cfg_cache: DynamicCache(batch=B', seq_len=L)
  -> cfg_logits: [B', L, S, V]
        |
        v
combined_cache = cat(cond_cache, cfg_cache) along batch
  -> DynamicCache(batch=2B', seq_len=L)

同时返回 cfg_logits（用于对齐 precomputed_logits 的最后一步）

如果 precomputed_logits 存在，还会这样拼一下让 batch 对齐：

precomputed_logits: [B', 1, S, V]
cfg_logits_last:   [B', 1, S, V]  (取 cfg_logits[:, -1:])
cat -> [2B', 1, S, V]


⸻

2) CFG 推理循环（每一步一步采样）——核心图

下面是每个 step 在你的代码里发生的事（for step in range(max_step)）：

(输入)
prev_token: [B', 1, S]    cache: batch = 2B' (如果 cfg>1)
        |
        | if cfg > 1:
        v
prev_token = tile(2)  -> [2B', 1, S]
        |
        +------------------------------+
        |  (A) 如果 precomputed_logits  |
        |      logits = precomputed    |
        |      shape: [2B',1,S,V]      |
        |                              |
        |  (B) 否则正常 forward:        |
        v                              |
_step(input_ids=prev_token, past_key_values=combined_cache, mask=mask?)
  -> logits_raw: [2B', 1, S, V]
  -> cache 更新（仍然 batch=2B'）
        |
        v
logits_cond, logits_uncond = logits_raw.chunk(2)
  logits_cond:   [B', 1, S, V]
  logits_uncond: [B', 1, S, V]
        |
        v
CFG 混合（你代码里的公式）:
mixed = logits_cond * cfg + logits_uncond * (1 - cfg)
     = logits_uncond + cfg * (logits_cond - logits_uncond)
shape: [B', 1, S, V]
        |
        v
masked_fill_(mask)  (只允许该 modality 的 token)
        |
        v
min_step 限制（可选）：强行禁止 EOS/EOT
        |
        v
_logit_to_token(temperature, topk)
prev_token(next): [B', 1, S]
        |
        v
追加到 hypos 列表，判断是否 EOS/EOT 结束

✅ 这就是 CFG 的核心：一次 forward 得到 cond/uncond 两套 logits（靠 batch 翻倍并行），然后做线性组合。

⸻

3) 收尾：把 cache 从 2B’ 还原回 B’（只保留条件分支）

推理结束后：

if cfg > 1:
  indices = arange(B')  # 只取前半（conditional）
  cache.batch_select_indices(indices)
  -> cache batch 回到 B'

然后它还会做一个“prefill last token”的 _step（为了多段生成衔接）：

prev_token[..., 1:] = 0   # 只保 stream0 的 token
_step(input_ids=prev_token, past_key_values=cache)
-> last_logits: [B',1,S,V]


⸻

一句话总览（最关键的三步）

(1) 造 unconditional cache：用全 0 token 跑出 cfg_cache
(2) 把 cond_cache 与 cfg_cache 拼成 batch=2B，并把每步输入 token 也 tile(2)
(3) logits 拆两半 -> mixed = uncond + cfg*(cond-uncond) -> 采样


⸻

如果你愿意，我还可以把这张图再“升级”成两张更直观的版本：
	1.	cond/uncond 两条流水线并行图（像电路图那种），
	2.	把 mask（modality 限制）插在图里，解释为什么 CFG 混合后还要再 mask 一次。

## CFG Graph v2

好，我们来两张“结构化图”。
我会用更直观的 并行流水线图 + 带 mask 位置标注图，完全对应你代码里的真实执行顺序。

⸻

图 1️⃣：Conditional / Unconditional 并行流水线图（CFG 核心结构）

这是最核心的一张 —— 展示两条分支如何在 batch 维并行计算。

⸻


                 ┌─────────────────────────────┐
                 │         Prefill 阶段         │
                 │  input_ids: [B, T0, S]       │
                 └──────────────┬──────────────┘
                                │
                                ▼
                        Transformer Forward
                                │
                                ▼
                    cache_cond (batch=B, len=T0)
                                │
                ─────────────────────────────────
                                │
                                ▼
                   ┌───────────────────────┐
                   │  CFG 预处理阶段       │
                   │  (_prepare_cfg_cache) │
                   └─────────┬─────────────┘
                             │
         zeros input         │
  [B, T0, S] 全 0 token       │
             │               │
             ▼               ▼
     Transformer         Transformer
     (conditional)       (unconditional)
        │                     │
        ▼                     ▼
   cache_cond             cache_uncond
   batch=B                batch=B
                             │
                             ▼
               ┌────────────────────────┐
               │  拼接 batch 维         │
               │  batch = 2B            │
               └──────────┬─────────────┘
                          ▼
              combined_cache (2B, T0)


⸻

🚀 进入每一步生成 step：

prev_token: [B,1,S]

if cfg>1:
    prev_token = tile(2)
    shape -> [2B,1,S]

                │
                ▼
        Transformer Forward
        with combined_cache
                │
                ▼
      logits_raw: [2B,1,S,V]
                │
                ▼
     ┌───────────────────────────────┐
     │ 拆分 batch 维 (chunk(2))      │
     └──────────────┬────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
 logits_cond            logits_uncond
 [B,1,S,V]              [B,1,S,V]
         │                     │
         └──────────┬──────────┘
                    ▼
        CFG 混合公式：
        mixed = uncond + cfg*(cond - uncond)
                    │
                    ▼
            mixed_logits [B,1,S,V]
                    │
                    ▼
               采样 token


⸻

⭐ 关键理解

CFG 在这里不是“多跑一次模型”，
而是：

把两条分支塞进 batch 维，一次 forward 同时算完

这是最省算力、最标准的实现方式。

⸻

⸻

图 2️⃣：加入 Modality Mask 的完整推理图

你这个模型是多模态 + 多 stream，所以 mask 非常重要。

下面是带 mask 的完整版本。

⸻

Step 内完整流程

prev_token
    │
    ▼
(if cfg) tile(2)
    │
    ▼
Transformer Forward
    │
    ▼
logits_raw [2B,1,S,V]
    │
    ▼
chunk -> cond / uncond
    │
    ▼
CFG 混合
    │
    ▼
mixed_logits [B,1,S,V]
    │
    ▼
──────────────────────────────
        🔴 关键：Mask 在这里
──────────────────────────────
logits.masked_fill_(mask, -inf)

mask shape: [1,1,S,V]
含义：
  - 只允许当前 modality 的 token
  - 禁止其他 modality 的 vocab 区间
  - stream 1+ 只允许对应 interval
  - stream 0 允许 <|eot|>, <|eos|>
──────────────────────────────
    │
    ▼
(可选) min_step 抑制 EOS/EOT
    │
    ▼
topk + temperature 采样
    │
    ▼
next_token


⸻

🎯 为什么 CFG 混合后还要再 mask？

非常关键的一点：
	•	CFG 可能会把某些“非法 token”拉高
	•	但多模态系统中每个 stream 只能用自己那段 vocab interval
	•	所以必须最后再 mask 一次，强制合法

否则：
	•	audio segment 里可能采样到 text token
	•	text segment 里可能采样到 audio code

⸻

⸻

🧠 两张图总结一句话

图 1 讲的是：

CFG 是 batch 维拼接 + logits 线性插值

图 2 讲的是：

多模态系统必须在 CFG 后再加一次 vocab mask

⸻

⸻

如果你愿意，我可以再画一张：

🔥「CFG + 多 stream + interval softmax」三者叠加的完整大图

那张图会把：
	•	stream 维度
	•	vocab interval
	•	CFG batch doubling
	•	mask 逻辑

全部合并成一张“终极结构图”。