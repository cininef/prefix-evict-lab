# prefix-evict-lab 图解：从 AI infra 概念到每一行代码

这份文档从零开始讲这个项目：先讲大模型推理里的基础概念，再按阶段（M1 → M6）讲每一步要解决什么问题、代码怎么写、得到了什么结论。

每一节都对应仓库里的真实代码（摘录有删减，以文件为准）。每个阶段在 git 里都有 tag，可以用 `git checkout m3-latency-model` 回到当时的状态。

---

## 目录

- [第 0 部分：AI infra 基础概念](#第-0-部分ai-infra-基础概念)
  - [0.1 大模型推理分两步：prefill 和 decode](#01-大模型推理分两步prefill-和-decode)
  - [0.2 KV cache 是什么](#02-kv-cache-是什么)
  - [0.3 衡量推理快慢的指标：TTFT 和 TPOT](#03-衡量推理快慢的指标ttft-和-tpot)
  - [0.4 显存是瓶颈：PagedAttention 和分块](#04-显存是瓶颈pagedattention-和分块)
  - [0.5 前缀缓存（prefix caching）](#05-前缀缓存prefix-caching)
  - [0.6 缓存满了怎么办：淘汰策略](#06-缓存满了怎么办淘汰策略)
- [第 1 部分：项目要回答的问题和整体结构](#第-1-部分项目要回答的问题和整体结构)
- [M1：模拟器](#m1模拟器)
- [M2：接上真实模型，验证正确性](#m2接上真实模型验证正确性)
- [M3：测量真实延迟，建立延迟模型](#m3测量真实延迟建立延迟模型)
- [M4：更真实的 workload](#m4更真实的-workload)
- [M5：自适应淘汰策略](#m5自适应淘汰策略)
- [M6：在真实引擎上逐个策略测量 TTFT](#m6在真实引擎上逐个策略测量-ttft)
- [方法论：这个项目里学到的做实验的规矩](#方法论这个项目里学到的做实验的规矩)
- [附录：术语表、怎么运行、文件地图](#附录)

---

## 第 0 部分：AI infra 基础概念

"AI infra"（AI 基础设施）在这里指的是**让大模型跑得快、跑得省**的那一层系统：显存怎么管、请求怎么调度、计算结果怎么复用。这个项目研究的是其中很小但很关键的一块：**前缀缓存的淘汰策略**。要理解它，需要先理解下面几个概念。

### 0.1 大模型推理分两步：prefill 和 decode

大语言模型（LLM）一次只生成一个 token（大致相当于一个词或半个词）。处理一个请求分两个阶段：

```
用户的 prompt: "请解释什么是 KV cache"          (假设是 8 个 token)

阶段 1：prefill（预填充）
  一次性把 8 个 token 全部送进模型，并行计算
  ──> 得到第 1 个输出 token "KV"

阶段 2：decode（解码）
  每次只送入上一步生成的 1 个 token
  "KV"    ──> "cache"
  "cache" ──> "是"
  "是"    ──> ...                        (一直到结束)
```

- **prefill** 是**计算密集型**的：prompt 越长，要做的矩阵乘法越多。几千个 token 的 prompt，prefill 可能要几百毫秒到几秒；
- **decode** 是**访存密集型**的：每一步只算 1 个 token，但要把整个模型的权重从显存里读一遍。

这个项目关心的是 **prefill**：能不能少算一点？

### 0.2 KV cache 是什么

Transformer 的核心是 **attention（注意力）**：每个 token 都要"看"它前面所有的 token。具体来说，每个 token 在每一层都会算出三个向量：

- **Q（query）**：我在找什么；
- **K（key）**：我是什么（供别人匹配）；
- **V（value）**：我携带的信息。

第 i 个 token 的输出 = 用它的 Q 去和第 0..i 个 token 的 K 做匹配，再按匹配程度对它们的 V 加权求和。

关键观察：**第 i 个 token 的 K 和 V 只取决于第 0..i 个 token**（因为是因果注意力，看不到后面的内容）。所以算过的 K、V 可以存起来，后面的 token 直接复用，不需要重算。这些存下来的张量就叫 **KV cache**。

```
层 1:  K[0] K[1] K[2] ... K[i]      V[0] V[1] V[2] ... V[i]
层 2:  K[0] K[1] K[2] ... K[i]      V[0] V[1] V[2] ... V[i]
...
层 24: ...

每个 K[j]、V[j] 的形状：[kv_heads, head_dim]
```

decode 阶段之所以能"每次只送入 1 个 token"，就是因为前面所有 token 的 K、V 都在 KV cache 里。

KV cache 很占显存。以本项目用的 Qwen2.5-0.5B 为例：24 层，2 个 KV head，每个 head 64 维，fp32 精度，每个 token 需要 `24 层 × 2(K和V) × 2 头 × 64 维 × 4 字节 ≈ 24.6 KB`。一个 2000 token 的 prompt 大约要 49 MB。大模型、多个并发请求叠加起来，KV cache 经常比模型权重还大。

### 0.3 衡量推理快慢的指标：TTFT 和 TPOT

- **TTFT（Time To First Token）**：从请求到达到第一个 token 输出所用的时间，主要由 **prefill** 决定。用户感受到的"响应速度"就是它；
- **TPOT（Time Per Output Token）**：之后每个 token 的间隔，由 decode 决定。

前缀缓存优化的是 **TTFT**：复用 KV cache，就能跳过一部分 prefill。

### 0.4 显存是瓶颈：PagedAttention 和分块

早期的推理系统给每个请求预留一整段连续的显存来放 KV cache，按最大可能长度预留，浪费很大。

**PagedAttention**（vLLM 提出）借鉴了操作系统的**虚拟内存分页**：
- 把 KV cache 切成固定大小的**块（block）**，例如每块 16 个 token；
- 显存里有一个**块池（block pool）**，请求需要时才分配一块；
- 每个请求维护一张"块表"，记录自己的第 1、2、3……块分别在池子里的哪个位置。

```
块池（物理显存）:  [块0][块1][块2][块3][块4][块5][块6][块7] ...
请求 A 的块表:      块3 → 块0 → 块6          (逻辑上连续，物理上不连续)
请求 B 的块表:      块1 → 块4
空闲:               块2, 块5, 块7
```

本项目完全沿用这个设计：`block_allocator.py` 就是那个块池，块大小是 16 个 token。

### 0.5 前缀缓存（prefix caching）

很多请求的**开头是一样的**：

```
Agent 多轮对话：
  第 1 轮: [system prompt][工具说明][用户问题1]
  第 2 轮: [system prompt][工具说明][用户问题1][回答1][工具输出1][用户问题2]
  第 3 轮: [第 2 轮的全部内容 ............................][回答2][问题3]
           └──────── 和上一轮完全相同的前缀 ─────────┘

RAG（检索增强生成）：
  请求 1: [system prompt][文档 A][文档 B][问题1]
  请求 2: [system prompt][文档 A][文档 C][问题2]
           └─── 相同 ───┘└相同┘
```

既然第 i 个 token 的 K、V 只取决于它前面的 token，**两个请求只要前缀一字不差，这段前缀的 KV cache 就完全一样**。前缀缓存就是：把算过的前缀的 KV cache 留在块池里，新请求先查一下最长有多少前缀已经缓存了，**只对剩下的部分做 prefill**。

- **SGLang** 的 RadixAttention 用一棵**基数树（radix tree）**来组织缓存的前缀；
- **vLLM** 的 Automatic Prefix Caching 用对块内容做哈希的方式实现同样的效果。

一个重要的限制：**只能复用前缀**。只要中间有一个 token 不同，后面的内容即使字面完全相同也不能复用，因为它们"看到"的上文不一样，K、V 也就不一样。这个限制在 RAG 场景下会变得很关键（见 M4）。

### 0.6 缓存满了怎么办：淘汰策略

块池的大小是固定的。缓存满了以后，新请求要放进来，就必须**淘汰（evict）**一些旧的块。淘汰谁，决定了能保住多少前缀缓存的收益。

经典的淘汰策略（来自 CPU 缓存、操作系统的页面置换、数据库的缓冲池）：

| 策略 | 规则 | 依据 | 弱点 |
|---|---|---|---|
| **LRU**（Least Recently Used） | 淘汰最久没被访问的 | 最近用过的，很可能马上还会用 | 一波一次性的访问（扫描）会把常用数据冲掉 |
| **LFU**（Least Frequently Used） | 淘汰访问次数最少的 | 一直常用的，以后也会常用 | 过去很热、现在没人用的数据赖着不走 |
| **ARC**（Adaptive Replacement Cache） | 同时跟踪"最近"和"频率"，根据自己犯的错调整偏向 | 自适应 | 实现更复杂 |
| **Belady（oracle）** | 淘汰下一次被用到的时间最晚的 | 需要预知未来，理论最优 | 现实中无法实现，只作为上限参照 |

前缀树上的淘汰还有一个特殊约束：**只能淘汰叶子节点**。如果淘汰了中间的块，它下面的块就失去了前缀，变得没有意义。

---

## 第 1 部分：项目要回答的问题和整体结构

**研究问题：** 在 agent 多轮对话和 RAG 这类 workload 下，前缀缓存能省下多少 prefill？块池容量有限时，这些收益能保住多少？不同的淘汰策略差多少？换算成真实的 TTFT 又差多少？

**整体结构：**

```
                 ┌────────────────────────────────────────────────┐
  workload       │ trace.py         合成的 agent 多轮对话          │
  生成器         │ branch_trace.py  带子 agent 分叉、工具等待      │
                 │ rag_trace.py     RAG（合成 Zipf / 真实检索）    │
                 └──────────────────────┬─────────────────────────┘
                                        │ 一串 prompt（token id 列表）
                                        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ simulator.py：对每个请求                                          │
 │   查最长缓存前缀 → 锁住 → 插入没缓存的块 → 解锁                    │
 │   统计：命中了多少 token、淘汰了多少次                             │
 └───────────────┬────────────────────────────────┬─────────────────┘
                 ▼                                ▼
   ┌───────────────────────────┐    ┌──────────────────────────────────┐
   │ radix_cache.py            │    │ policies.py / oracle.py          │
   │ 块粒度的前缀树             │◄───│ LRU, LFU, CostAware, ARC,        │
   │ 引用计数（防止误删）        │    │ ARCAged, Belady                  │
   │ 只淘汰叶子                 │    └──────────────────────────────────┘
   └─────────────┬─────────────┘
                 ▼
   ┌───────────────────────────┐
   │ block_allocator.py        │  固定大小的块池
   └───────────────────────────┘

 engine.py   真实引擎：同一棵前缀树 + 真实的 Qwen 模型 + 真实的 KV 张量
 latency.py  从真实引擎测出的延迟模型：把命中率换算成 TTFT
```

**为什么要同时有模拟器和真实引擎？**
- 模拟器很快：几秒钟就能重放几千个请求，可以扫很多参数（策略 × 缓存大小 × 随机种子）；
- 但模拟器只能给出"命中率"。命中率高 10 个百分点，TTFT 能快多少毫秒？必须在真实模型上测；
- 所以先用真实引擎测出一个**延迟模型**，再用它把模拟器的命中率换算成 TTFT，最后在真实引擎上抽查，验证换算是准的。

---

## M1：模拟器

> tag：`m1-simulation`
> 目标：用模拟器比较淘汰策略，并给出理论上限。

### 1.1 块池：`block_allocator.py`

最底层是一个固定大小的块池，相当于显存里的"物理页框"。它只负责分配和回收块号，不关心块里存了什么：

```python
class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))   # 空闲块号
        self._used: set[int] = set()

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks            # 池子满了，调用方需要先淘汰
        block = self._free.pop()
        self._used.add(block)
        return block

    def free(self, block: int) -> None:
        self._used.remove(block)
        self._free.append(block)
```

### 1.2 前缀树：`radix_cache.py`

这是整个项目的核心数据结构。

**每个节点 = 一个块 = 16 个 token。** 从根到某个节点的路径，就是一段被缓存的前缀：

```
prompt A: [t0..t15][t16..t31][t32..t47]
prompt B: [t0..t15][t16..t31][u32..u47]      (前 32 个 token 和 A 相同)

            root
              │
         [t0..t15]          ← 块 3（A、B 共享）
              │
         [t16..t31]         ← 块 0（A、B 共享）
           ┌──┴──┐
   [t32..t47]  [u32..u47]   ← 块 6（只属于 A）、块 1（只属于 B）
```

节点的定义：

```python
@dataclass(eq=False)
class Node:
    key: tuple             # 这个块的 16 个 token id
    parent: "Node | None"
    block: int = -1        # 在块池里的块号
    depth: int = 0
    children: dict         # key -> 子节点
    ref_count: int = 0     # 正在使用它的请求数；> 0 时不能被淘汰
    last_access: int = 0   # 最近一次被访问的时间（LRU 用）
    hit_count: int = 0     # 被命中的次数（LFU 用）
    created: int = 0
    seq: int = 0           # 插入顺序，优先级相同时用来打破平局
    pid: int = ROOT_PID    # 前缀 id（M5 加入，见后文）
```

**查询最长缓存前缀**（`match_prefix`）：把 prompt 切成 16 个 token 一块，沿着树往下走，直到走不下去：

```python
def match_prefix(self, tokens) -> list[Node]:
    self._clock += 1                       # 每个请求时钟加 1
    node, path = self.root, []
    for chunk in self._chunks(tokens):     # 按 16 个 token 切块
        child = node.children.get(chunk)
        if child is None:
            break                          # 前缀到这里断了
        child.last_access = self._clock    # 更新访问统计，供淘汰策略使用
        child.hit_count += 1
        path.append(child)
        node = child
    return path                            # 命中的节点路径
```

注意：只缓存**完整的块**。prompt 末尾不满 16 个 token 的部分不会被缓存。

**插入新的块**（`insert_nodes`）：从命中路径的末尾开始，为剩下的每一块分配一个新节点。池子满了就先淘汰：

```python
def insert_nodes(self, tokens, matched):
    chunks = self._chunks(tokens)
    node = matched[-1] if matched else self.root
    fresh = []
    for chunk in chunks[len(matched):]:
        ...
        block = self._allocate_block()       # 满了会触发 evict_one()
        if block is None:
            break                            # 实在没有能淘汰的了
        child = Node(key=chunk, parent=node, block=block, ...)
        node.children[chunk] = child
        child.ref_count += 1                 # 先锁住，防止被本次插入自己淘汰
        fresh.append(child)
        node = child
    self.unlock(fresh)
    return fresh
```

`child.ref_count += 1` 这一行背后有一个真实的 bug 故事：早期版本没有锁住刚插入的块，导致 LFU 这类按访问次数淘汰的策略会把**本请求刚插入的块**（访问次数为 0）立刻淘汰掉。LFU 看起来比 LRU 差很多，直到修复这个 bug 才恢复正常。回归测试是 `test_insert_does_not_evict_its_own_new_blocks`。

**引用计数（`lock` / `unlock`）：** 一个请求在使用某段缓存的前缀时，要先把它锁住（`ref_count += 1`），用完再解锁。被锁住的块不能被淘汰，否则请求读到一半，KV 就被别的请求覆盖了。

**淘汰（`evict_one`）：** 只在"没有被锁住的叶子"中挑选，由策略决定淘汰哪一个：

```python
def evict_one(self) -> bool:
    leaves = [n for n in self._leaves if n.ref_count == 0]
    if not leaves:
        return False
    victim = min(leaves, key=lambda n: (self.policy.priority(n, self._clock), n.seq))
    parent = victim.parent
    del parent.children[victim.key]
    self._leaves.discard(victim)
    if not parent.children and parent is not self.root:
        self._leaves.add(parent)          # 父节点变成了新的叶子
    self.allocator.free(victim.block)     # 块还给池子
    ...
```

`self._leaves` 是 M5 期间加的优化：原来每次淘汰都要遍历整棵树找叶子，在 RAG 的长 prompt 上占了 99% 的运行时间。改成增量维护叶子集合后快了约 190 倍，结果完全一致。

### 1.3 淘汰策略：`policies.py`

策略的接口非常简单：**给每个节点算一个优先级，优先级最低的先被淘汰。**

```python
class LRU:
    name = "lru"
    def priority(self, node, now):
        return (node.last_access,)                   # 越久没访问，越先淘汰

class LFU:
    name = "lfu"
    def priority(self, node, now):
        return (node.hit_count, node.last_access)    # 访问越少越先淘汰，平局看最近访问

class CostAware:
    def priority(self, node, now):
        cost = 1.0 + 0.1 * node.depth                # 猜测：越深的块重算越贵
        return ((node.hit_count + 1) * cost, node.last_access)
```

新增一个策略只需要写一个小类。

### 1.4 理论上限：`oracle.py`（Belady）

Belady 算法能"看到未来"：先扫描整个 trace，记下每个前缀在哪些请求里会被用到；淘汰时选**下一次被用到的时间最晚**的块：

```python
def priority(self, node, now):
    uses = self._uses.get(node.pid, [])      # 这个前缀出现在哪些请求里（有序）
    pos = bisect_left(uses, now)             # 二分查找：下一次使用在哪里
    next_use = uses[pos] if pos < len(uses) else NEVER
    return (-next_use,)                      # 越晚用到，优先级越低，越先淘汰
```

因为只能淘汰叶子，它不是严格意义上的最优解，而是一个很强的参照上限。

### 1.5 模拟器和合成 trace：`simulator.py`、`trace.py`

合成的 agent trace：几个共享的 system prompt；40 个 session 各自进行 6 轮对话，每轮在历史后面追加 64 个 token；所有 session 随机交错，模拟服务器同时服务很多对话。

模拟器的主循环就是"查、锁、插、解锁"：

```python
def simulate(trace, cache):
    for tokens in trace:
        matched = cache.match_prefix(tokens)
        cache.lock(matched)
        cache.insert(tokens, matched)
        cache.unlock(matched)
        hit += len(matched) * cache.block_size     # 命中的 token 数
        ...
```

### 1.6 M1 的结论

10 个随机种子，误差棒是一个标准差：

| 块数 | LRU | LFU | Belady |
|---:|---:|---:|---:|
| 64  | 0.407 | 0.508 | 0.538 |
| 128 | 0.575 | 0.578 | 0.705 |
| 256 | 0.698 | 0.642 | 0.807 |
| 512 | 0.814 | 0.755 | 0.859 |

1. **缓存小的时候 LFU 好，缓存大的时候 LRU 好。** system prompt 被所有 session 共享，访问频率高；每个 session 最新的尾部是私有的，只看最近访问。没有一个信号能同时照顾两者；
2. **离理论上限还差最多约 13 个百分点**，有改进空间。

---

## M2：接上真实模型，验证正确性

> tag：`m2-real-model-parity`
> 目标：证明"复用缓存的 KV"和"从头计算"得到的结果**完全一样**。

### 2.1 为什么要先做这一步

模拟器默认了一个前提：复用 KV 等价于重新计算。如果这个前提不成立，命中率越高，模型输出反而越错，后面所有结论都没有意义。所以必须在真实模型上验证：**同一个 prompt，走缓存和不走缓存，生成的 token 必须逐个相同**（贪心解码，即每步都取概率最高的 token）。

### 2.2 KV 块池：`engine.py` 里的 `KVPool`

真实引擎需要一个真正存 KV 张量的池子。关键设计是：**它和前缀树用同一套块号**。

```python
class KVPool:
    """K 和 V 的形状都是 [层数, 块数, kv_heads, 块大小, head_dim]"""

    def __init__(self, config, num_blocks, block_size, dtype, device):
        shape = (config.num_hidden_layers, num_blocks, kv_heads, block_size, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
```

这样，前缀树里的淘汰策略淘汰了块 6，真实显存里的块 6 就被释放，可以给别的请求用。**模拟器里的策略可以原封不动地跑在真实模型上**，这是 M6 能做端到端对比的前提。

**gather（取出）**：把命中的若干块，按顺序拼接成 HuggingFace 模型认识的 `DynamicCache` 格式：

```python
def gather(self, blocks):
    cache = DynamicCache()
    idx = torch.tensor(blocks, device=self.k.device)
    for layer in range(self.k.shape[0]):
        # [n块, 头数, 16, 维度] -> [1, 头数, n*16, 维度]
        k = self.k[layer, idx].permute(1, 0, 2, 3).flatten(1, 2).unsqueeze(0)
        v = self.v[layer, idx].permute(1, 0, 2, 3).flatten(1, 2).unsqueeze(0)
        cache.update(k, v, layer)
    return cache
```

**scatter（写回）**：模型算完之后，把新的完整块的 K、V 拷贝回池子里对应的块号：

```python
def scatter(self, cache, start_block, blocks):
    for layer, cl in enumerate(cache.layers):
        for i, b in enumerate(blocks):
            lo = (start_block + i) * self.block_size
            self.k[layer, b] = cl.keys[0, :, lo : lo + self.block_size]
            self.v[layer, b] = cl.values[0, :, lo : lo + self.block_size]
```

### 2.3 一个请求的完整流程：`PrefixEngine.generate`

```python
def generate(self, prompt, max_new_tokens=16):
    bs = self.cache.block_size
    matched = self.cache.match_prefix(prompt)            # 1. 查最长缓存前缀
    matched = matched[: (len(prompt) - 1) // bs]          #    至少留 1 个 token 不复用（见下）
    self.cache.lock(matched)                              # 2. 锁住，防止被淘汰
    try:
        n_cached = len(matched) * bs
        past = self.pool.gather([n.block for n in matched])   # 3. 从池里取出 KV
        suffix = torch.tensor([prompt[n_cached:]])
        out = self.model(input_ids=suffix,                    # 4. 只对后缀做 prefill
                         past_key_values=past, use_cache=True)

        fresh = self.cache.insert_nodes(prompt, matched)      # 5. 新的完整块写回池子
        if fresh:
            self.pool.scatter(out.past_key_values, len(matched),
                              [n.block for n in fresh])

        nxt = int(out.logits[0, -1].argmax())                 # 6. 贪心解码
        ...
    finally:
        self.cache.unlock(matched)                            # 7. 解锁
```

**两个容易出错的边界情况：**

1. **整个 prompt 都命中时，也必须留 1 个 token 重新计算。** 下一个 token 的预测来自最后一个位置的输出（logits），不做前向计算就拿不到这个输出。所以最多复用 `(len-1)//16` 块，最后一块一定重算；
2. **重算的那一块不能再存一份。** 原来的 `insert` 假设命中路径之后的节点都不存在，会为已经存在的块再建一个节点，旧块就泄漏了。修复方法是：遇到已经存在的子节点就沿着它往下走，并临时锁住：

```python
existing = node.children.get(chunk)
if existing is not None:
    existing.ref_count += 1          # 临时锁住，防止它在本次插入中被淘汰
    walked.append(existing)
    node = existing
    continue
```

### 2.4 测试

`tests/test_engine_parity.py` 用一个**随机初始化的小 Qwen2 模型**（不需要下载，CI 里也能跑）覆盖四种情况：冷请求、共享 system prompt、整个 prompt 都命中、只有 10 块的池（强制淘汰）：

```python
def test_parity_under_eviction_pressure(model):
    eng = make_engine(model, 10)                      # 远小于工作集
    for _ in range(12):
        p = rng.choice(systems) + toks(rng, rng.randrange(5, 30))
        r = eng.generate(p, NEW)
        assert r.tokens == reference(model, p)        # 必须和 HF generate 逐 token 相同
    assert eng.cache.num_evictions > 0                # 确认真的发生了淘汰
```

淘汰压力下的测试最重要：块被反复复用时，最容易出现"读到别人的 KV"这类错误。

`benchmarks/real_model_parity.py` 在真实的 Qwen2.5-0.5B-Instruct 上用三段交错的多轮对话验证：**6/6 回复逐 token 相同**，池子缩小到 14 块（发生淘汰）时也一样。

### 2.5 一个坑：参照本身是错的

第一次在真实模型上跑，**连没有任何缓存的请求都对不上**。说明问题不在缓存，而在参照。原因是 Qwen-Instruct 的 `generation_config` 自带 `repetition_penalty=1.1`，而 transformers 5 的 `generate` 即使设置了 `do_sample=False` 也会把它混进来。解决方法是显式关掉所有采样参数：

```python
greedy = dict(do_sample=False, max_new_tokens=args.new, repetition_penalty=1.0,
              top_k=None, top_p=None, temperature=None)
```

教训：**验证一个系统之前，先确认参照是对的。** 这里是靠第三个参照（手动不用缓存、逐步取 argmax）来判断到底哪一边错了。

---

## M3：测量真实延迟，建立延迟模型

> tag：`m3-latency-model`
> 目标：命中率不等于 TTFT。测出"缓存了 C 个 token、还要算 N 个新 token 时，prefill 要多久"。

### 3.1 延迟模型的形式：`latency.py`

prefill 的耗时由两部分组成：

- **线性部分**：每个新 token 都要过一遍投影层和 MLP，正比于 N；
- **注意力部分**：N 个新 token 中的每一个，都要看前面所有的 C 个缓存 token，平均还要看一半的其他新 token，所以正比于 `N × (C + N/2)`。

再加上两项真实测量中发现的成本：

- **地板（floor）**：在 GPU 上，算的 token 很少时，耗时几乎是个常数（见 3.3）；
- **gather 成本**：把命中的块从池子里拷出来也要时间，这是命中带来的额外开销，不计算它就会高估缓存的收益。

```
prefill(C, N) = max(floor, a + b·N + c·N·(C + N/2))
gather(C)     = g0 + g1·C              （C = 0 时为 0）
TTFT(C, N)    = gather(C) + prefill(C, N)
```

代码：

```python
@dataclass
class LatencyModel:
    a: float; b: float; c: float
    g0: float = 0.0; g1: float = 0.0
    floor: float = 0.0

    def prefill(self, cached, new):
        lin = self.a + self.b * new + self.c * new * (cached + new / 2)
        return max(self.floor, lin)

    def gather(self, cached):
        return self.g0 + self.g1 * cached if cached else 0.0

    def ttft(self, cached, new):
        return self.gather(cached) + self.prefill(cached, new)
```

拟合用最小二乘法，为了让核心包不依赖 numpy，用纯 Python 解正规方程。

有了这个模型，模拟器的每个请求都能换算成预计的 TTFT：

```python
def estimated_ttft(trace, per_request_hit, model):
    out = []
    for tokens, hit in zip(trace, per_request_hit):
        cached = min(hit, len(tokens) - 1)        # 和引擎一样，至少留 1 个 token
        out.append(model.ttft(cached, len(tokens) - cached))
    return out
```

### 3.2 怎么测：`benchmarks/prefill_latency.py`

1. 先对整个 prompt 做一次前向计算，把 KV 写进池子；
2. 对每个 (总长度, 缓存长度) 组合：从池子里 gather 缓存的块，只对后缀做前向，分别计时。这和引擎走的是同一条路径；
3. 用 5×5 的网格点拟合，另外随机抽 12 个点作为**留出验证集**，只看训练集上的 R² 不够。

### 3.3 测量中踩过的坑（每一个都是因为某项检查没通过才发现的）

**坑 1：误差棒小得不真实。** 最初对每个点连续重复 5 次，四分位距（IQR）只有 1–3 ms，看起来很精确。但同一个计算，过一分钟再测就慢了 13%。连续 5 次处在同一个"机器状态"下，所以彼此很像；漂移会变成某些点的系统偏差。

**修复：打乱顺序分轮测量。** 每一轮把所有点打乱顺序各测一次，取跨轮次的中位数：

```python
for r in range(warmup_rounds + rounds):
    order = list(points)
    rng.shuffle(order)                       # 每轮打乱
    for total, cached in order:
        pt, gt = sample(model, pool, prompt[:total], cached)
        ...
```

**坑 2：端到端预测偏高 13%。** 依次排除了几个假设：
- GPU 温度？满载 3 分钟前后耗时没变化（291 ms 对 294 ms），❌；
- 大请求之后紧跟小请求会变慢？37.9 ms 对 38.3 ms，❌；
- **主机 CPU 负载？** 用 10 个进程占满所有 CPU 核：32-token 的小请求慢了 20%，1024-token 的大请求只慢了 3%，✅。

原因是：GPU 上小请求的耗时主要花在 **CPU 端逐个提交 kernel**（24 层的每个算子都要由 Python/CPU 提交给 GPU）上，而不是 GPU 计算本身。这正是"地板"的来源：大约 38 ms，不管算多少个 token。测量那次正好主机比较忙，于是现在会把 load average 记录下来。

**坑 3：地板只存在于 GPU 上。** CPU 上计算和分发用的是同一组核，没有"GPU 在空等分发"的区间，所以加地板反而更不准（短后缀误差：地板模型 11.3%，纯线性 8.9%）。现在按设备决定是否使用地板。

### 3.4 M3 的结论

- MPS（Mac GPU）：拟合 R²=0.998，留出点平均误差 7%；用真实引擎跑 120 个请求，**平均 TTFT 预测误差只有 +3.5%**，引擎和模拟器的命中数完全一致；
- **TTFT 的收益有天花板**：38 ms 的地板意味着，即使每个请求都全部命中，平均 TTFT 也只能降到无缓存时的约 29%；
- **中等缓存大小下，策略差距会转化成实打实的延迟**：128 块时 Belady 比 LRU 的命中率高 13 个百分点，TTFT 低 19%；
- **重算成本几乎不随深度变化**：每个块的相对成本约为 `1 + 0.004 × 深度`，而 M1 里 CostAware 猜的是 `1 + 0.1 × 深度`，高估了 25 倍。所以淘汰决策的价值主要来自**预测复用**，而不是预测重算成本。

---

## M4：更真实的 workload

> tag：`m4-workloads`
> 目标：M1 只用了一种合成 trace。换成更多样的 workload，结论还成立吗？

### 4.1 子 agent 分叉和工具等待：`branch_trace.py`

真实的 agent 有两个 M1 trace 没有的特点：

1. **分叉**：从同一段长上下文分出多个子任务，子任务共享一个很深的前缀；
2. **工具等待**：发出工具调用后，要等工具返回才会发下一轮，这段时间它的尾部块完全空闲。

用事件驱动来模拟，每个 session 有一个 `ready_at`（什么时候可以发下一轮）：

```python
s.ready_at = now + (rng.expovariate(1 / cfg.pause_mean)        # 以一定概率进入工具等待
                    if rng.random() < cfg.pause_prob else 0.0)
...
if s.is_root and not s.forked and cfg.fanout > 0:              # 根 agent 跑完几轮后分叉
    for _ in range(cfg.fanout):
        sessions.append(_Session(list(s.history), cfg.child_turns, parent=sid, ...))
```

**一个推翻了直觉的发现：** 我原本以为工具等待会让复用距离变长，从而伤害缓存，于是写了测试来断言这一点，结果**测试失败了**：平均复用距离反而从 14.2 降到了 10.1。

原因：session 总数固定，每个请求总会分给某个 session，所以**平均复用距离大致等于并发的 session 数**，工具等待改变不了它。工具等待改变的是**分布的形状**：没在等待的 session 循环得更快（中位数变小），在等待的 session 空闲更久（长尾变长）。也就是流量变得更"突发"了。这反而**提高了命中率**，即使把负载对齐之后也是如此。测试最后改成了断言分布的离散程度。

### 4.2 RAG：`rag_trace.py`

RAG 的 prompt 是 `system prompt + k 篇检索到的文档 + 问题`。根据 0.5 节的前缀限制，**文档的排列顺序决定了能复用多少**：

```python
if cfg.order == "canonical":
    picked.sort()                                  # 按文档 id 排序
elif cfg.order == "popular_first":
    picked.sort(key=lambda d: rank_of[d])          # 热门文档排在前面
# 否则保持检索器给出的顺序
```

**真实检索数据：`benchmarks/build_real_rag.py`。** 合成数据假设文档热度服从 Zipf 分布。为了校准，用公开数据集 BEIR NFCorpus（3633 篇文档、3237 条查询），对每条查询用 BM25 检索出 top-5，用 Qwen 的 tokenizer 计算长度。只保存结构（文档 id、长度、检索结果），因为对缓存来说，token 的具体内容不重要，重要的是"哪些文档相同"和"有多长"。

### 4.3 M4 的结论

1. **没有一个固定策略能通吃。** agent 类 workload（普通、分叉、工具等待）上，128 块以上 LRU 最好；RAG 上，LFU 在每个缓存大小下都最好；
2. **RAG 上文档顺序比淘汰策略重要得多。** 无限大缓存下的命中率上限：

   | 文档顺序 | 合成数据 | 真实 NFCorpus |
   |---|---:|---:|
   | 检索器给的顺序 | 0.432 | 0.305 |
   | 按文档 id | 0.474 | 0.384 |
   | 热门优先 | 0.571 | 0.427 |

   只是改变拼接 prompt 的顺序，不碰淘汰策略，上限就能提高 12–14 个百分点；
3. **真实检索数据的复用比合成数据少得多。** 真实热度分布的尾部又长又平（Zipf 指数约 0.54，90% 的文档至少被检索到一次），合成数据高估了文档复用。

---

## M5：自适应淘汰策略

> tag：`m5-adaptive-policy`
> 目标：既然 agent 偏好 LRU、RAG 偏好 LFU，能不能设计一个自动适应的策略？

### 5.1 基础设施：前缀 id 和策略钩子

ARC 需要记住"最近被淘汰的块"，所以每个块需要一个**稳定的身份**：同一条 token 路径，被淘汰后再插入，身份应该不变。用链式哈希实现：

```python
def chain_pid(parent_pid, key):
    return hash((parent_pid, key))     # 前缀 id = hash(父前缀 id, 本块的 token)
```

这样每个节点只需要在创建时算一次（O(1)）。Belady 也改用它，原来每次都要把整条路径拼起来再哈希，改完之后快了 10 倍。

ARC 是有状态的，需要知道插入、命中和淘汰事件，所以给策略接口加了几个**可选的钩子**：

```python
class EvictionPolicy(Protocol):
    def priority(self, node, now): ...
    # 可选：
    # on_insert(node, now)   插入了新块
    # on_hit(node, now)      某个块被命中
    # on_evict(node, now)    某个块被淘汰
    # choose(leaves, now)    直接选出要淘汰的叶子，替代"取最小优先级"
```

缓存在对应的位置检查策略有没有这些方法，有就调用。原来的 LRU、LFU 完全不需要改。

### 5.2 ARC 适配到前缀树

```
T1：缓存了但还没被复用过的块      （偏"最近"）
T2：被复用过至少一次的块          （偏"频率"）
B1、B2：最近从 T1、T2 淘汰的块的 pid（"幽灵列表"，只记身份，不占显存）
p：T1 的目标大小
```

**自适应的核心**：如果一个刚被淘汰的块又被请求了，说明淘汰错了。

```python
def on_insert(self, node, now):
    if node.pid in self.b1:          # 从 T1 淘汰的块又回来了 → T1 太小了
        self.p = min(self.c, self.p + max(1.0, len(self.b2) / len(self.b1)))
        del self.b1[node.pid]
        self.t2.add(node)
    elif node.pid in self.b2:        # 从 T2 淘汰的块又回来了 → T2 太小了
        self.p = max(0.0, self.p - max(1.0, len(self.b1) / len(self.b2)))
        del self.b2[node.pid]
        self.t2.add(node)

def on_hit(self, node, now):
    self.t2.add(node)                # 第一次被复用：从 T1 升到 T2

def choose(self, leaves, now):
    t1 = [n for n in leaves if n not in self.t2]
    t2 = [n for n in leaves if n in self.t2]
    preferred = t1 if (self.live - len(self.t2)) > self.p else t2
    pool = preferred or t1 or t2     # 只能淘汰叶子：想要的那类没有可淘汰的，就退而求其次
    return min(pool, key=lambda n: (n.last_access, n.seq))
```

### 5.3 诊断 ARC 最弱的地方

ARC 在 agent trace、64 块时落后 LFU 4.3 个百分点。先后检验了两个假设：

- **假设 1：叶子约束导致 ARC 常常选不到想淘汰的那类。** 加计数器测量"被迫退回"的比例：64 块时 21.4%，256 块时 20.4%，几乎一样，而 ARC 在 256 块时表现很好。❌ 否定；
- **假设 2：system prompt 块被错误淘汰。** 3 个 system prompt 共占 48 块，缓存总共只有 64 块。ARC 的 T2 内部只按最近访问时间排序，被命中上百次的 system prompt 块和只被复用过一次的块是平等的。统计三种策略淘汰 system prompt 块的次数：

  | 64 块 | 命中率 | 淘汰的 system prompt 块 |
  |---|---:|---:|
  | LRU | 0.408 | 5608 |
  | ARC | 0.464 | 3366 |
  | LFU | 0.510 | 1468 |

  排序和命中率完全对应。✅ 证实。

### 5.4 修复：ARCAged

让 T2 内部也考虑频率。但直接用累计命中次数会重新引入 LFU 的老问题（过去很热、现在不用的块赖着不走）：试过之后，小缓存提升了 2.6 个百分点，大缓存却下降了 2.3，不采用。

最终方案：**频率随时间衰减**。每个块维护一个分数，每次命中时先衰减再加 1，半衰期为 20 个请求：

```python
def _decayed(self, node, now):
    sc, t = self.score.get(node, (0.0, now))
    return sc * 2.0 ** (-(now - t) / self.half_life)     # 指数衰减

def on_hit(self, node, now):
    super().on_hit(node, now)
    self.score[node] = (self._decayed(node, now) + 1.0, now)

def choose(self, leaves, now):
    ...
    if pool is t2:   # T2 内部：先按衰减分数的对数分桶，再按最近访问
        return min(pool, key=lambda n: (int(self._decayed(n, now) + 1).bit_length(),
                                        n.last_access, n.seq))
    return min(pool, key=lambda n: (n.last_access, n.seq))
```

一直被访问的 system prompt 分数保持在高位；已经结束的 session 分数会逐渐衰减。半衰期越长（越接近 LFU），大缓存时的损失越大：agent 512 块时，半衰期 20、50、200 分别落后 0.3、0.7、1.6 个百分点。这个"剂量-反应"关系也印证了"过时频率"这个机制。

### 5.5 怎么公平地评估：`benchmarks/paired.py`

- **配对比较**：同一个随机种子下，不同策略跑的是同一条 trace，所以逐个种子计算差值，比看两个误差棒是否重叠更有说服力；
- **严格的基准**：在每个 (workload, 缓存大小) 格子里，和"那个格子里最好的固定策略"比较。这是事后才知道的，在线运行时不可能提前选对，所以这个基准对自适应策略是偏严格的；
- **防止过拟合**：半衰期 20 是在 3 个 workload 上挑出来的，所以必须在**没有参与调参**的 workload 上验证。

结果（50 个格子）：

| 策略 | 平均落后（百分点） | 最坏落后 |
|---|---:|---:|
| LRU | 2.51 | 10.07 |
| LFU | 0.72 | 5.87 |
| CostAware | 1.82 | 7.26 |
| **ARCAged** | **−0.40**（平均略好于事后最佳） | **2.29** |

在没参与调参的真实 NFCorpus 数据上，15 个格子里 ARCAged **每一个都在全部 5 个种子上**胜过 LFU。和普通 ARC 相比，它是严格的改进：没有任何一个格子明显变差。

但要诚实地说：提升只有约 1 个百分点，而 Belady 还领先 5–13 个百分点。**ARCAged 的价值在于稳健，而不是大幅提升。**

---

## M6：在真实引擎上逐个策略测量 TTFT

> tag：`m6-e2e-ttft`
> 目标：前面的 TTFT 都是用延迟模型算出来的。直接在真实引擎上测，结论还成立吗？

### 6.1 实验设计：`benchmarks/e2e_policies.py`

- agent trace，120 个请求；LRU、LFU、ARCAged × 64/128/256 块 × 3 个随机种子；
- 每个请求只生成 1 个 token，这样测到的就是 TTFT；
- 吸取 M3 的教训：**所有组合打乱顺序运行**，避免漂移偏向某个策略；测量期间不运行任何其他任务，并记录 load average：

```python
runs = [(p, b, s) for p in policies for b in sizes for s in range(args.seeds)]
random.Random(0).shuffle(runs)                       # 打乱运行顺序
for p, b, s in runs:
    sim = simulate(trace, RadixCache(b, BS, POLICIES[p]()))
    pred = st.mean(estimated_ttft(trace, sim.per_request_hit, lm))     # 预测
    eng = PrefixEngine(model, RadixCache(b, BS, POLICIES[p]()))
    for tokens in trace:
        r = eng.generate(tokens, 1)                                     # 实测
        ...
```

### 6.2 结果

相对 LRU 的 TTFT 变化，实测（括号内为预测）：

| 块数 | LFU | ARCAged |
|---:|---:|---:|
| 64  | −8.7%（−7.3%） | −3.1%（−3.8%） |
| 128 | +1.1%（+3.5%） | −2.2%（−2.0%） |
| 256 | +8.0%（+9.8%） | −1.4%（−0.9%） |

- **LFU 的表现会翻转**：缓存小时最快，缓存大时最慢；
- **ARCAged 在三种缓存大小下都比 LRU 快**；
- 每次运行的平均 TTFT 和预测值相差都在 4.4% 以内，策略排名完全一致，引擎和模拟器的命中数在每次运行中都完全相同。

这闭合了整个链条：**模拟器 → 延迟模型 → 真实引擎**，三者给出一致的结论。

局限：只有 3 个种子；一次只处理一个请求；0.5B 的小模型；跑在笔记本的 GPU 上；只测了 agent trace。

---

## 方法论：这个项目里学到的做实验的规矩

1. **先验证正确性，再谈性能。** M2 在所有性能实验之前；
2. **验证系统之前，先验证参照。** 冷请求都对不上，说明是参照有问题（repetition penalty）；
3. **误差棒要反映真实的波动。** 连续重复测量的误差棒会骗人，要打乱顺序分轮测量；
4. **用留出数据验证。** 训练集上的 R² 不够；调过参的策略必须在没见过的 workload 上测；
5. **配对比较优于看误差棒是否重叠。**
6. **一个假设没通过，就去找真正的机制。** 本项目里被否定的假设：GPU 温度、大请求干扰小请求、叶子约束、"工具等待伤害缓存"。每一次否定都把结论推得更准确；
7. **单个随机种子的结论不可靠。** canonical 顺序在单个种子上看起来没用，10 个种子平均后是正的；
8. **把负面结果也写下来。** CostAware 没用、命中次数排序不采用、工具等待不伤害缓存，这些都写在提交记录和 README 里。

---

## 附录

### 术语表

| 术语 | 含义 |
|---|---|
| token | 模型处理文本的最小单位，大致相当于一个词或半个词 |
| prefill | 一次性处理整个 prompt 的阶段，计算密集 |
| decode | 逐个生成输出 token 的阶段，访存密集 |
| KV cache | 每层 attention 的 Key、Value 张量，存下来供后续 token 复用 |
| TTFT | Time To First Token，第一个输出 token 的延迟 |
| block（块） | KV cache 的分配单位，本项目中是 16 个 token |
| prefix caching | 跨请求复用相同前缀的 KV cache |
| hit rate（命中率） | prompt 中从缓存复用的 token 所占比例 |
| eviction（淘汰） | 缓存满了时移除一些块 |
| oracle / Belady | 能预知未来的理论最优淘汰策略，作为上限参照 |
| workload / trace | 一串请求序列，用来测试系统 |
| seed（随机种子） | 控制随机生成的 trace，不同 seed 给出不同但同分布的 trace |
| MPS | Apple 芯片上的 GPU 后端 |

### 怎么运行

```bash
pip install -e ".[dev]"
pytest                                                    # 44 个测试

python benchmarks/multi_seed.py                           # M1：多 seed + oracle

pip install -e ".[dev,model]"                             # 需要 torch + transformers
python benchmarks/real_model_parity.py                    # M2：正确性
python benchmarks/prefill_latency.py --device mps         # M3：测量 + 拟合延迟
python benchmarks/validate_latency_e2e.py docs/latency_mps.json --device mps
python benchmarks/ttft_from_sim.py docs/latency_mps.json  # 命中率 → TTFT

python benchmarks/workload_sweep.py docs/latency_mps.json # M4：agent workload
python benchmarks/rag_order.py docs/latency_mps.json      # M4：RAG 文档顺序
python benchmarks/build_real_rag.py                       # M4：真实检索数据
python benchmarks/paired.py docs/workloads_*.json docs/rag_order_*.json --candidate arc_aged   # M5
python benchmarks/e2e_policies.py docs/latency_mps.json --device mps                         # M6
```

### 文件地图

| 文件 | 作用 | 阶段 |
|---|---|---|
| `src/prefixlab/block_allocator.py` | 固定大小的块池 | M1 |
| `src/prefixlab/radix_cache.py` | 块粒度前缀树、引用计数、淘汰 | M1（M5 优化） |
| `src/prefixlab/policies.py` | LRU、LFU、CostAware、ARC、ARCAged | M1、M5 |
| `src/prefixlab/oracle.py` | Belady 理论上限 | M1 |
| `src/prefixlab/simulator.py` | 重放 trace、统计命中 | M1 |
| `src/prefixlab/trace.py` | 合成 agent 多轮对话 | M1 |
| `src/prefixlab/engine.py` | 真实模型引擎：KV 池、只算后缀的 prefill | M2 |
| `src/prefixlab/latency.py` | 延迟模型：命中率 → TTFT | M3 |
| `src/prefixlab/branch_trace.py` | 子 agent 分叉、工具等待 | M4 |
| `src/prefixlab/rag_trace.py` | RAG（合成 Zipf / 真实检索） | M4 |
| `benchmarks/paired.py` | 配对比较、最大遗憾 | M5 |
| `benchmarks/e2e_policies.py` | 真实引擎上逐策略测 TTFT | M6 |
| `docs/*.json`、`docs/*.png` | 各阶段的原始数据和图 | 各阶段 |
