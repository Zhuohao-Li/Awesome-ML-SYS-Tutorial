# veRL Multi-Turn RL 工作流程详解

## 概览

我们以 veRL 进行 multi-turn RL 的完整工作流程作为例子，帮助读者更好理解 veRL 各组件的工作流程。在这个文档中，我们使用 SGLang rollout 和 FSDP 训练引擎的集成来进行讲述

## 整体架构

### 核心组件

| 组件 | 文件路径 | 主要功能 |
|------|----------|----------|
| RayPPOTrainer | `verl/trainer/ppo/ray_trainer.py` | 主训练协调器 |
| ActorRolloutRefWorker | `verl/workers/fsdp_workers.py:95` | Hybrid Engine |
| SGLangRollout | `verl/workers/rollout/sglang_rollout/sglang_rollout.py:219` | SGLang rollout |
| CriticWorker | `verl/workers/fsdp_workers.py:1443` | critic function估计 |
| RewardModelWorker | `verl/workers/fsdp_workers.py:1134` | reward model |
| / | `verl/trainer/main_ppo.py` | 训练程序入口 |

RayPPOTrainer是整个训练的中央控制器，协调所有worker的工作，实现计算、控制流分离。

ActorRolloutRefWorker负责执行rollout的全过程，生成experience。

## 详细工作流程

### 1. 初始化阶段

veRL 训练系统的启动是一个精心设计的多阶段过程。系统需要协调多个分布式组件，包括训练引擎（FSDP）、推理引擎（SGLang）、奖励计算模块等。这个阶段的核心任务是建立起一个可扩展的分布式训练环境。

#### 1.1 程序启动
**文件**: `verl/trainer/main_ppo.py:158-179`

**为什么需要这个步骤？**
在分布式RL训练中，我们需要创建一个中央协调器来管理所有的计算资源和训练流程。RayPPOTrainer 就是这个协调器，它不直接参与计算，而是负责调度和协调各个 worker 的工作。

```python
# 创建 RayPPOTrainer 实例
trainer = RayPPOTrainer(
    config=config,                    # 包含所有超参数和配置信息
    tokenizer=tokenizer,              # 用于文本编码/解码
    processor=processor,              # 处理多模态数据（图像、视频等）
    role_worker_mapping=role_worker_mapping,  # 定义每种角色使用什么Worker类
    resource_pool_manager=resource_pool_manager,  # 管理GPU资源分配
    ray_worker_group_cls=ray_worker_group_cls,    # Ray分布式通信的具体实现
    # ...
)
trainer.init_workers()  # 初始化 workers，跳转到1.2
trainer.fit()          # 开始训练
```

**关键概念解释：**
- **role_worker_mapping**: 这是veRL架构的核心概念。系统将不同的功能分配给不同角色的Worker：
  - `ActorRollout`: 负责推理生成和Actor训练
  - `Critic`: 负责价值函数估计（仅GAE算法需要）
  - `RewardModel`: 负责奖励模型计算（optional）
  - `RefPolicy`: 负责reference策略计算（用于KL散度）

#### 1.2 Worker 初始化
**文件**: `verl/trainer/ppo/ray_trainer.py:693-787`

**为什么需要Worker Group？**
在大规模训练中，我们需要将模型分布到多个GPU上。Worker Group 是对一组执行相同任务的Worker的抽象。比如，一个ActorRollout WorkerGroup可能包含8个Worker，每个Worker运行在不同的GPU上，通过 rank_id 来indexing，这些workers共同完成一个batch的推理生成任务。

```python
def init_workers(self):
    # 1. 创建资源池 - 这步骤告诉Ray我们有哪些GPU资源可用
    self.resource_pool_manager.create_resource_pool()
    
    # 2. 为每个角色创建对应的Worker类定义
    # RayClassWithInitArgs是一个包装器，它记住了Worker的初始化参数
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[Role.ActorRollout],  # 实际的Worker类
        config=self.config.actor_rollout_ref,            # 这个角色的配置
        role="actor_rollout",                            # 角色标识符
    )
    
    # 3. 创建WorkerGroup并初始化模型
    # WorkerGroup会在多个GPU上启动多个Worker实例
    self.actor_rollout_wg = all_wg["actor_rollout"]
    self.actor_rollout_wg.init_model() # 初始化actor model
```

**数据流向说明：**
```
配置文件 → RayPPOTrainer → ResourcePool → WorkerGroup → 多个Worker实例(GPU1, GPU2, ...)
```

#### 1.3 模型和引擎初始化
**文件**: `verl/workers/fsdp_workers.py:479-575`

**Hybrid Engine的设计理念：**
veRL采用"Hybrid Engine"设计，即在同一个Worker中集成训练引擎（FSDP）和推理引擎（SGLang）。这样设计的好处是：
1. **内存效率**：避免在不同引擎间复制模型权重
2. **通信效率**：减少进程间通信开销
3. **资源利用**：在推理和训练间动态切换GPU资源

```python
def init_model(self):
    # 1. 构建 FSDP actor 模型 - 这是用于训练的引擎
    # FSDP 将模型参数分片到多个GPU
    self.actor_module_fsdp, self.actor_optimizer = self._build_model_optimizer(...)
    
    # 2. 构建 SGLang rollout 引擎 
    if self._is_rollout:
        self.rollout, self.rollout_sharding_manager = self._build_rollout(...)
    
    # 3. 初始化其他组件（critic, reference policy 等）
    # 这些组件根据算法需要条件性初始化
```

**关键技术点：**
- **Sharding Manager**: 负责在FSDP训练格式和SGLang推理格式之间转换模型权重
    - 对于FSDP训练格式：每张 GPU 只保存自己那一片参数（shard），其余分片存在其他 GPU。参数通常是一个扁平的大张量，便于通信（all-gather / reduce-scatter）。适合梯度回传和参数更新，但单张 GPU 无法直接拿到完整模型做推理。
    - SGLang 推理格式，推理时需要拿到完整的权重（或者按 tensor-parallel 的方式重新切片），且要按 flash-attention 兼容的结构组织。必须是连续的权重张量，且通常要转换精度（bfloat16 → fp16）和布局（row-major → col-major）才能高效推理。
- **条件初始化**: 只有当前Worker负责某个功能时才初始化对应的模型组件

### 2. 单步训练流程

每一步训练包含四个核心阶段：数据加载→序列生成→经验处理→模型更新。这个流程体现了强化学习的核心思想：通过与环境交互获得经验，然后用这些经验改进策略。

#### 2.1 数据加载与预处理
**文件**: `verl/trainer/ppo/ray_trainer.py:915-970`

**为什么要分离生成数据？**
在RL训练中，输入数据分为两部分：用于生成的提示（prompts）和用于训练的完整序列。生成阶段只需要prompts，而训练阶段需要完整的prompt+response。这种分离让我们可以灵活处理不同长度的序列，并支持多模态输入。

```python
for batch_dict in self.train_dataloader:
    # DataProto是veRL的核心数据结构，统一处理tensor和非tensor数据
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    
    # 提取生成所需的数据 - 这些是"干净"的输入，不包含标准答案
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],  # 基础文本数据
        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "tools_kwargs"]  # 多模态和工具数据
    )
```

**数据结构解释：**
- **batch_keys**: PyTorch张量，可以直接送入模型
- **non_tensor_batch_keys**: 元数据，如图像、工具配置等
- **DataProto**: veRL设计的统一数据容器，支持复杂的批处理操作

#### 2.2 序列生成阶段 (Rollout)
**文件**: `verl/trainer/ppo/ray_trainer.py:946-954`

**什么是Rollout？**
Rollout是强化学习中的核心概念，指让当前策略与环境交互，生成一系列的状态-动作-奖励序列。在LLM训练中，"环境"就是语言任务本身，"动作"是生成下一个token，"状态"是当前的上下文。

这个阶段用来处理multi-turn对话的推理请求、tool use，最后生成响应sequence的candidates：

```python
# 调用 SGLang 进行序列生成
with _timer("gen", timing_raw):  # 性能监控包装器
    if not self.async_rollout_mode:
        # 同步模式：等待所有生成完成再继续
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # sync生成
    else:
        # 异步模式：生成和其他计算可以重叠进行
        self.async_rollout_manager.wake_up()
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch) # async生成
        self.async_rollout_manager.sleep()
```

**同步 vs 异步模式：**
- **同步模式**: 简单可靠，适合调试和小规模训练
- **异步模式**: 性能更高，inference和reward计算可以pipeline，适合大规模训练

**生成实现**: `verl/workers/fsdp_workers.py:627-662`

`generate_sequences()`函数的实现如下：

```python
def generate_sequences(self, prompts: DataProto):
    # 设置生成配置 - 这些参数控制生成的行为
    meta_info = {
        "eos_token_id": self.generation_config.eos_token_id,  # 结束符
        "pad_token_id": self.generation_config.pad_token_id,  # 填充符
    }
    
    # 权重格式转换：FSDP训练格式 → SGLang推理格式
    # 这是混合引擎的关键技术，确保两个引擎使用相同的模型权重
    with self.rollout_sharding_manager:
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
```

**关键技术细节：**
- **Sharding Manager**: 处理不同并行策略间的权重转换
- **上下文管理器**: 确保权重转换的原子性，避免不一致状态

**SGLang 多轮生成**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:889-1000`

**为什么需要异步请求级生成？**
多轮对话中，每个对话可能有不同的轮数和工具调用次数。传统的批处理方法会等待最慢的对话完成，造成资源浪费。异步请求级生成允许每个对话独立进行，显著提高并发效率。

```python
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # 支持多轮对话的异步请求级生成
    if self._tp_rank == 0:  # 只在主进程中调度请求
        # 将批处理数据转换为独立请求
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n=1 if is_validate else self.config.n,  # 验证时只生成1个候选，训练时生成n个
        )
        
        # 使用asyncio并发处理所有请求
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(*[
                self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) 
                for req in req_list
            ])
        ) # 最后生成候选的response list
```

**并发处理的优势：**
- **资源利用率高**: 不需要等待最慢的请求
- **支持复杂交互**: 每个对话可以有不同的工具调用序列
- **内存效率**: 避免为所有可能的轮数预分配内存

#### 2.3 经验处理阶段 (Make Experience)

**什么是Experience？**
在强化学习中，experience指的是智能体与环境交互产生的数据，包括状态、动作、奖励、下一状态等。在LLM训练中，experience包括输入prompt、生成的response、获得的reward，以及用于策略更新的各种概率和优势值。

ActorRolloutRefWorker通过`generate_sequences`方法生成训练的experience，核心功能是将prompts转化为experience加入到context中，并且计算对应的reward值

**重新计算 Log Probabilities**
**文件**: `verl/trainer/ppo/ray_trainer.py:995-1030`

**为什么要重新计算log_prob？**
在生成阶段，我们得到的log_prob是用于采样的，不够精确（通常是fp16/bf16），这是因为在rollout我们的目标是"快"，快速生成old_log_prob。而在训练阶段，我们需要更精确的概率计算用于PPO等RL算法，如果ratio = exp(new_log_prob - old_log_prob)差别太大，就会导致clip和KL约束失真，从而破坏训练的稳定性。此外，生成和训练可能使用不同的数值精度或计算图，重新计算确保一致性。

```python
# 重新计算 old_log_probs - 这是PPO算法的关键输入
with _timer("old_log_prob", timing_raw):
    # 使用当前策略重新计算生成序列的概率
    # 这个概率将作为PPO中的"old policy"概率
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(old_log_prob)  # 将结果合并到batch中
```

以PPO为例，PPO需要比较新旧策略的概率比值来控制更新幅度，防止策略变化过大导致训练不稳定。

**reward compute**
**文件**: `verl/trainer/ppo/ray_trainer.py:980-993`

**奖励计算的两种方式：**
veRL 支持灵活的奖励计算策略，可以结合 model-based reward 和 rule-based reward，适应不同类型的任务需求。

```python
with _timer("reward", timing_raw):
    # 方式1：奖励模型计算（可选）- 使用训练好的奖励模型评估响应质量
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    # 方式2：规则/函数奖励计算 - 使用任务特定的评估函数
    if self.config.reward_model.launch_reward_fn_async:
        # 异步计算：适合耗时的奖励函数（如代码执行、数学验证）
        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
    else:
        # 同步计算：适合简单快速的奖励函数
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

**奖励设计的考虑：**
- **模型奖励**: 通用但可能有偏差，适合对话、摘要等主观任务
- **规则奖励**: 准确但覆盖有限，适合数学、代码等客观任务
- **混合奖励**: 结合两者优势，在实际应用中很常见

**reference策略评估**
**文件**: `verl/trainer/ppo/ray_trainer.py:1031-1038`

**为什么需要参考策略？**
参考策略（Reference Policy）通常是训练前的原始模型，用于计算KL散度约束，防止新策略偏离原始行为太远。这在RLHF中特别重要，确保模型在获得高奖励的同时保持合理的语言行为。

```python
if self.use_reference_policy:
    with _timer("ref", timing_raw):
        if not self.ref_in_actor:
            # 独立的参考策略模型 - 需要额外的GPU内存，但更精确
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch) # 计算reference model的log_prob
        else:
            # 使用Actor中的LoRA基模型作为参考 - 内存高效，适合LoRA微调
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

**LoRA优化：**
当使用LoRA（Low-Rank Adaptation）微调时，可以将LoRA的基模型直接用作参考策略，大大节省内存。

**advantage函数计算**
**文件**: `verl/trainer/ppo/ray_trainer.py:1062-1076`

**什么是Advantage？**
Advantage函数衡量某个动作相对于平均水平的好坏程度。正值表示该动作比平均好，负值表示比平均差。这是策略梯度算法的核心，帮助模型学习更好的行为。

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,    # 优势估计方法：GAE、GRPO等
    gamma=self.config.algorithm.gamma,                    # 折扣因子：未来奖励的权重
    lam=self.config.algorithm.lam,                        # GAE参数：偏差vs方差的权衡
    num_repeat=self.config.actor_rollout_ref.rollout.n,   # 每个prompt的采样数
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,     # GRPO特殊归一化
    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable, # 多轮对话支持
    config=self.config.algorithm,
)
```

**不同优势估计方法：**
- **GAE**: 使用价值函数估计，偏差小但需要额外的Critic网络
- **GRPO**: 使用群组统计，无需价值函数，适合数学推理任务
- **REINFORCE**: 直接使用奖励，简单但方差大

#### 2.4 模型更新阶段 (Train)

**训练阶段的设计理念：**
veRL 采用分阶段训练策略。在训练初期，可以只训练 Critic 来稳定价值估计，然后再开始 Actor 训练。这种预热策略在复杂任务中特别有效。

**Critic 更新** (仅GAE算法)
**文件**: `verl/trainer/ppo/ray_trainer.py:1078-1083`

**为什么Critic需要单独更新？**
Critic网络负责估计状态价值，这个估计的准确性直接影响优势函数的质量。通过独立更新Critic，可以使用更多的训练步骤和不同的学习率，提高价值估计的准确性。

```python
if self.use_critic:  # 只有GAE算法需要Critic
    with _timer("update_critic", timing_raw):
        # Critic更新使用回归损失，目标是准确预测累积奖励
        critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

**Critic训练细节：**
- **损失函数**: 通常使用MSE回归损失
- **目标值**: 实际获得的累积奖励
- **更新频率**: 可能比Actor更频繁，以提高估计准确性

**Actor 更新**
**文件**: `verl/trainer/ppo/ray_trainer.py:1085-1092`

**Critic预热机制：**
在训练初期，Critic的价值估计可能不准确，导致Actor更新的方向错误。通过设置预热期，让Critic先稳定下来，再开始Actor训练。

```python
if self.config.trainer.critic_warmup <= self.global_steps:  # 预热期检查
    with _timer("update_actor", timing_raw):
        # 添加多轮对话的元信息，用于特殊处理
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        # 执行PPO策略更新
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

**PPO更新过程：**
1. 计算新旧策略的概率比值
2. 使用advantage加权策略梯度
3. 应用PPO的clip机制防止更新过大
4. 可选地添加KL散度惩罚

**FSDP Actor 更新实现**
**文件**: `verl/workers/fsdp_workers.py:575-625`

**内存管理的复杂性：**
大模型训练的一个挑战是内存管理。veRL支持参数卸载（offloading），将暂时不用的参数移到CPU或存储，需要时再加载回GPU。这个过程需要精心协调以避免性能损失。

```python
def update_actor(self, data: DataProto):
    # 数据预处理：移动到CPU以支持各种硬件配置
    data = data.to('cpu')
    
    # 内存优化：按需加载模型参数
    if self._is_offload_param:
        # 将FSDP模型参数从CPU/存储加载到GPU
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        # 将优化器状态加载到GPU
        load_fsdp_optimizer(optimizer=self.actor_optimizer, 
                           device_id=get_torch_device().current_device())
    
    # 执行实际的策略更新
    output = self.actor.update_policy(data=data)
```

**FSDP的优势：**
- **内存效率**: 参数分片减少单GPU内存需求
- **计算效率**: 避免不必要的参数复制
- **可扩展性**: 支持任意数量的GPU

## 多轮对话特殊处理

### 配置文件
**文件**: `examples/sglang_multiturn/config/tool_config/`

多轮对话需要特殊的工具配置和状态管理：

1. **工具集成**: 支持计算器、搜索等工具调用
2. **状态保持**: SGLang 引擎维护多轮对话状态
3. **掩码处理**: 使用 `loss_mask` 进行多轮损失计算
4. **异步处理**: 支持并发的多轮对话生成

### 关键实现
**文件**: `verl/trainer/ppo/ray_trainer.py:129-178`

```python
def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl", multi_turn=False):
    if multi_turn:
        # 多轮对话使用 loss_mask 而不是 response_mask
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
```

## 性能优化点

### 1. 异步Rollout
**文件**: `verl/workers/rollout/async_server.py`
- 支持异步推理，提高GPU利用率
- 并发处理多个生成请求

### 2. 权重分片管理
**文件**: `verl/workers/sharding_manager/`
- 在推理和训练引擎间高效转换模型权重
- 减少内存占用和传输开销

### 3. 序列长度平衡
**文件**: `verl/trainer/ppo/ray_trainer.py:862-875`
- 自动平衡不同GPU间的token数量
- 提高训练效率

---

## GRPO 算法示例: 无价值函数的群组相对策略优化

### GRPO 工作原理

**文件**: `verl/trainer/ppo/core_algos.py:166-196`

GRPO (Group Relative Policy Optimization) 是一种无需价值函数的强化学习算法，特别适用于数学推理等需要多次采样的任务：

1. **群组采样**: 对每个prompt生成多个响应（如n=5）
2. **相对奖励**: 计算每组内响应的平均奖励作为基线
3. **优势计算**: `advantage = reward - group_mean`
4. **无Critic**: 不需要训练独立的价值网络

### 核心算法实现

**GRPO优势计算**: `verl/trainer/ppo/core_algos.py:166-196`

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """GRPO优势计算 - 基于群组相对奖励"""
    scores = token_level_rewards.sum(dim=-1)  # 序列级奖励
    
    # 按prompt索引分组计算统计量
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])
    
    # 计算每组的均值和标准差
    for idx in id2score:
        if len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor(id2score[idx]))
        else:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
    
    # 计算相对优势
    for i in range(scores.shape[0]):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
```

### 配置差异对比

#### PPO vs GRPO 关键配置

| 配置项 | PPO | GRPO |
|--------|-----|------|
| `algorithm.adv_estimator` | `gae` | `grpo` |
| `actor_rollout_ref.rollout.n` | `1` | `5` (或更大) |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | `true` |
| `algorithm.use_kl_in_reward` | `true` | `false` |
| Critic 组件 | **需要** | **不需要** |

#### 算法选择逻辑
**文件**: `verl/trainer/ppo/ray_trainer.py:323-346`

```python
if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
    self.use_critic = True  # 使用价值函数
elif self.config.algorithm.adv_estimator in [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.GRPO_PASSK,
    AdvantageEstimator.REINFORCE_PLUS_PLUS,
    # ...
]:
    self.use_critic = False  # 无价值函数
```

### GRPO 性能优势

1. **内存节省**: 无需训练Critic网络
2. **训练稳定**: 基于群组统计，减少方差
3. **适用场景**: 特别适合数学推理、代码生成等需要多次采样的任务
4. **实现简单**: 避免value function的复杂调参

### 高级扩展: DrGRPO

**文件**: `docs/algo/grpo.md`

DrGRPO 解决了GRPO的长度偏差问题：

```bash
# DrGRPO配置差异
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm  # 关闭序列维度平均
actor_rollout_ref.actor.use_kl_loss=false                      # 不使用KL loss
algorithm.norm_adv_by_std_in_grpo=false                        # 关闭标准差归一化
```

## 总结

veRL 的 VLM multi-turn RL 工作流程通过模块化设计实现了：

1. **高效推理**: SGLang 提供快速的多轮对话生成
2. **分布式训练**: FSDP 支持大模型的并行训练  
3. **灵活奖励**: 支持多种奖励函数组合
4. **异步处理**: 提高GPU利用率和训练效率
5. **多模态支持**: 原生支持视觉-语言模型
6. **算法多样性**: 支持PPO、GRPO、RLOO等多种RL算法

整个系统设计既保证了性能，又维持了良好的可扩展性和易用性。
