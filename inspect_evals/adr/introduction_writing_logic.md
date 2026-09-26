# Introduction 写作逻辑链

本文 Introduction 的核心任务，是把读者从一个普遍存在的 agent 执行问题，引向 ARA 的核心设计，并说明各个机制为何必要。整体逻辑可以概括为：

> 单个任务内部的工作具有不同执行粒度，而现有单智能体通常使用扁平控制流统一处理
> -> 现有规划与分解方法仍不能动态改变 agent 当前工作的作用域
> -> 因此需要一种可在当前作用域执行、必要时递归缩小作用域、完成后再返回原作用域的控制机制
> -> ARA 用动态执行栈、Execute/Decompose 路由和自适应局部批次实现这一机制
> -> Reduce 与 checkpoint compression 分别解决跨递归边界和作用域内部的状态管理
> -> 最后通过实验检验递归控制在不同任务难度和模型能力下的有效性与代价。

## 一、逐段论证结构

### 第 1 段：提出核心矛盾——任务内部异质性与扁平执行流不匹配

本段从 language-model agent 的通用背景切入：agent 通过长程推理、工具调用和环境交互解决任务。但同一个任务中的工作项并不具有相同的执行粒度：

- 有些工作已经足够明确，可以直接执行；
- 有些工作仍包含未解决的依赖、模糊目标或多个内部阶段，需要进一步组织。

现有单智能体系统通常把两类工作都放进同一条扁平交互轨迹中。这会产生两种相反的问题：

- 对过于宽泛的工作，agent 只能在局部 action loop 中隐式组织复杂结构；
- 对已经足够具体的工作，继续拆分会带来重复规划，却不产生有用结构。

因此，本段建立全文的出发点：**执行机制必须能够适应工作项的实际粒度。**

### 第 2 段：排除不充分解法——规划不等于作用域控制

本段依次分析几类常见方法的局限：

- Step-wise agent 能及时吸收新观察，但短决策跨度可能反复发现相同的局部结构；
- Plan-then-execute 能提供长程组织，但可能在前置条件和中间结果尚未知时过早承诺后续步骤；
- Upfront 或 fixed-depth decomposition 使用固定分解尺度，可能过度拆解可直接执行的工作，也可能对复杂工作拆解不足。

这些方法解决的是“下一步做什么”或“计划多远”，但没有显式解决“agent 应该在多大的作用域内工作”。由此引出真正缺失的能力：

1. 当前目标可执行时，留在当前作用域直接行动；
2. 当前目标过宽时，递归进入更窄的目标；
3. 子目标完成后，返回并恢复此前暂停的 continuation。

本段还指出，小模型更难仅靠隐式推理吸收宽泛且异质的工作项，因此更可能从显式作用域控制中获益。

### 第 3 段：给出核心方案——用单智能体递归控制实现动态作用域

本段首次完整定义 Adaptive Recursive Agents (ARA)：它不是多智能体系统，而是在传统单智能体交互循环上增加自适应递归控制。

ARA 的核心组成如下：

- 一个持续存在的 execution context；
- 一个由临时 execution frame 组成的动态栈；
- 所有 frame 均由同一个 controller 管理；
- local planner 在每个 frame 中同时输出可变长度的 work-item batch，以及每个 work item 的 `Execute`/`Decompose` 模式。

两种模式对应两类控制流：

- `Execute`：留在当前 frame，通过 ReAct-style interaction 直接处理；
- `Decompose`：暂停当前 frame，让同一个 controller 针对更窄目标递归调用；嵌套调用结束后恢复原 frame。

该段同时完成一个重要的概念澄清：递归调用是**同一执行过程中的临时作用域**，并非多个独立 agent；树状结构只是对嵌套调用轨迹的事后表示。

### 第 4 段：解释两个自适应维度——作用域与承诺跨度

本段把 ARA 的两个关键决策维度区分开：

- execution mode 决定**在哪里执行**：留在当前作用域，还是进入更窄作用域；
- batch length 决定**一次计划多远**：在获取新证据前承诺多少局部工作。

ARA 不把规划 horizon 固定为一步，也不直接覆盖整个剩余任务，而是根据当前信息只规划“下一个有用批次”。连续的 mode decision 在线构建执行结构，连续的 local planning call 则降低陈旧计划不断延伸的风险。

这一段的作用是说明：**递归控制负责组织层级，自适应局部规划负责控制每层中的推进节奏。** 两者互补，但不是同一个机制。

### 第 5 段：补齐状态闭环——递归返回与长轨迹压缩

仅能递归进入子目标还不够，系统还必须可靠地返回并管理不断增长的执行状态。本段引入两个机制：

#### Reflective Reduce：管理递归边界之间的状态

嵌套调用结束时，`Reduce` 将已完成 frame 压缩成 caller 所需的抽象结果，并检查该返回结果是否暴露出不完整、不一致或尚未解决的要求。如果存在问题，Reduce 会同时生成：

- caller-level result；
- localized repair work。

修复工作仍由同一个 controller 处理，完成后再恢复未受影响的 continuation。因此，Reduce 不只是摘要操作，还负责返回时的验证和局部修复。

#### Checkpoint compression：管理单个作用域内部的长执行

所有 frame 共享 persistent context，因此长时间的直接执行仍可能造成局部轨迹膨胀。ARA 在每经过 $K$ 次工具调用后，将尚未完成的 `Execute` item 压缩为 continuation state，并重置局部预算后继续执行。

两个机制的职责边界是：

- Reduce：处理 recursive return boundary；
- compression：处理 unfinished local execution 内部的状态增长。

### 第 6 段：从方法转向验证——提出实验问题

本段把前述方法主张转化为实验问题，计划验证：

- 递归控制是否提升 agentic task performance；
- 它在何种任务难度和模型能力条件下最有价值；
- adaptive local planning 相比 fixed 和 flat alternatives 是否更有效；
- 额外结构带来怎样的 computation 与 context trade-off。

当前正文在此处保留了结果 TODO。最终版本需要补充 benchmark/model coverage，以及两到三个最重要的定量结果，否则 Introduction 的逻辑链只有“问题 -> 方法 -> 验证目标”，还没有闭合为“问题 -> 方法 -> 实验证据 -> 结论”。

### 贡献列表：压缩并对应前文主张

三项贡献与前文机制一一对应：

1. **控制流贡献**：把 adaptive recursion 定义为单智能体控制流，使同一 controller 能在共享持久上下文中动态进入和退出临时执行作用域。
2. **规划与返回贡献**：提出 executability-guided routing、adaptive local batches 和 reflective Reduce，并在返回阶段直接生成局部修复工作。
3. **上下文管理与实证贡献**：为未完成的直接执行引入 checkpointed continuation，并研究递归控制与任务难度、模型能力、计算开销及上下文增长之间的关系。

## 二、逻辑链中的关键概念分工

| 概念 | 回答的问题 | 在逻辑链中的作用 |
| --- | --- | --- |
| `Execute` / `Decompose` | 当前工作是否可在现有作用域直接执行？ | 动态选择执行作用域 |
| Recursive frame | 如何暂停当前工作并进入更窄目标？ | 提供显式 call-and-return 控制流 |
| Adaptive batch length | 获取新信息前应规划多少局部工作？ | 平衡短视重规划与过早长程承诺 |
| Reflective `Reduce` | 子调用完成后应向 caller 返回什么？ | 抽象结果、检查缺口并触发局部修复 |
| Checkpoint compression | 当前直接执行过长时如何控制状态增长？ | 压缩未完成轨迹并延续执行 |
| Persistent context | 如何保证不同 frame 属于同一 agent 过程？ | 维持跨作用域的一致执行状态 |

## 三、全文论证闭环

Introduction 的完整论证可压缩为以下六步：

1. **观察**：单个任务内部存在不同粒度、不同可执行性的工作项。
2. **矛盾**：扁平 agent loop 用同一种控制方式处理所有工作项。
3. **缺口**：已有规划和固定分解可以改变计划内容或长度，却不能提供动态、显式的作用域切换与返回。
4. **方案**：ARA 通过 `Execute`/`Decompose` 决策、递归 frame 和自适应 batch 实现在线执行结构。
5. **闭环机制**：Reduce 保证递归调用可靠返回，checkpoint compression 控制作用域内部的长轨迹。
6. **验证**：实验分析性能收益、适用条件，以及计算和上下文代价；最终稿还需用定量结果完成证据闭环。

因此，Introduction 最核心的一句话不是“ARA 会分解任务”，而是：**ARA 让同一个 agent 根据工作项的可执行性，动态改变当前执行作用域，并以显式的 call-and-return 机制恢复原有工作。**
