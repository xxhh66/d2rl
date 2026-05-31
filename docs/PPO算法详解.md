# PPO 算法详解：从 Actor-Critic 到裁剪策略优化

> 本笔记以最常用的 **PPO-Clip（Clipped Proximal Policy Optimization）** 为主线，衔接 REINFORCE 与 Actor-Critic，解释概率比、裁剪目标、GAE、价值损失、熵奖励、训练流程与 PyTorch 核心实现。

---

## 1. PPO 是什么

PPO（Proximal Policy Optimization，近端策略优化）是 Schulman 等人在 2017 年提出的一类 on-policy 策略梯度算法。它的基本思想是：

> 使用当前旧策略采集一批交互数据，然后对策略进行若干轮小步更新；通过裁剪策略概率比，抑制新策略在一次迭代中偏离旧策略过远。

PPO 通常建立在 Actor-Critic 框架之上：

| 组件 | 学习对象 | 作用 |
|---|---|---|
| Actor | 策略 $\pi_\theta(a\mid s)$ | 生成动作分布 |
| Critic | 状态价值 $V_\phi(s)$ | 估计状态价值、辅助计算优势 |
| PPO 裁剪机制 | 新旧策略概率比 | 限制过度策略更新 |

PPO 论文提出了 PPO-Penalty 与 PPO-Clip 等形式；实际学习与工程实现中通常优先讲解 PPO-Clip。

---

## 2. 为什么 Actor-Critic 之后还需要 PPO

### 2.1 REINFORCE：直接用回报更新策略

$$
L_{\text{REINFORCE}}
=
-\sum_t G_t\log\pi_\theta(A_t\mid S_t)
$$

其问题是必须等待完整轨迹，并且蒙特卡洛回报方差较大。

### 2.2 Actor-Critic：用优势评价动作

Actor-Critic 用 Critic 提供优势估计：

$$
L_{\text{actor}}
=
-\hat{A}_t\log\pi_\theta(A_t\mid S_t)
$$

常见的一步优势信号是 TD 误差：

$$
\hat{A}_t
\approx
\delta_t
=
R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t)
$$

它降低了方差，并能更及时地更新策略。

### 2.3 普通 Actor-Critic 的剩余问题：更新可能过猛

假设旧策略在某状态下对一个动作的概率是：

$$
\pi_{\theta_{\text{old}}}(a\mid s)=0.20
$$

一次更新后，新策略将其变为：

$$
\pi_\theta(a\mid s)=0.90
$$

若该动作的优势估计带有噪声，这类激进更新可能使策略突然退化。另一方面，训练数据来自旧策略；新策略偏离旧策略越大，反复用旧策略数据优化新策略就越不可靠。

PPO 的目标就是：

> 保留 Actor-Critic 的优势学习机制，同时控制每轮策略改变幅度，使同一批 on-policy 数据可以安全地进行若干轮 minibatch 更新。

---

## 3. 旧策略、新策略与概率比

在一轮 PPO 更新中：

| 策略 | 符号 | 作用 |
|---|---|---|
| 旧策略 | $\pi_{\theta_{\text{old}}}$ | 与环境交互，采集本轮数据 |
| 新策略 | $\pi_\theta$ | 在已采集数据上被优化的策略 |

采样时必须保存旧策略对已执行动作的概率或对数概率：

$$
\log\pi_{\theta_{\text{old}}}(A_t\mid S_t)
$$

优化时重新计算新策略概率：

$$
\log\pi_\theta(A_t\mid S_t)
$$

二者构成策略概率比：

$$
r_t(\theta)
=
\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{\text{old}}}(A_t\mid S_t)}
$$

数值实现中通常使用对数概率差：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(A_t\mid S_t)
-
\log\pi_{\theta_{\text{old}}}(A_t\mid S_t)
\right)
$$

概率比的含义为：

| $r_t(\theta)$ | 含义 |
|---:|---|
| $r_t=1$ | 新策略未改变该动作概率 |
| $r_t>1$ | 新策略提高了该动作概率 |
| $r_t<1$ | 新策略降低了该动作概率 |

---

## 4. 未裁剪的代理目标

在旧策略数据上，可构造重要性采样形式的代理目标：

$$
L^{\text{CPI}}(\theta)
=
\mathbb{E}_t\left[r_t(\theta)\hat{A}_t\right]
$$

若 $\hat{A}_t>0$，则提高该动作概率、即增大 $r_t$，会使目标增大；若 $\hat{A}_t<0$，则降低该动作概率、即减小 $r_t$，会使目标增大。

问题在于，若完全不限制 $r_t$，优化器可能将某些动作概率改变得过多，导致新策略远离产生训练数据的旧策略。

---

## 5. PPO-Clip 核心目标函数

PPO-Clip 的最大化目标为：

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t
\right)
\right]
$$

其中：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
$$

把用于计算保守收益的概率比限制在：

$$
[1-\epsilon,\;1+\epsilon]
$$

之内。$\epsilon$ 常以 $0.1$ 或 $0.2$ 作为基础实验起点。

由于 PyTorch 等框架执行梯度下降，代码中使用负号定义策略损失：

$$
L_{\text{policy}}
=
-\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t
\right)
\right]
$$

---

## 6. 裁剪机制的四种情况

设：

$$
\epsilon=0.2
$$

安全区间为：

$$
[0.8,\;1.2]
$$

### 6.1 优势为正：这是一个值得增加概率的动作

若：

$$
\hat{A}_t=2>0
$$

| $r_t$ | $r_t\hat{A}_t$ | 裁剪项 | 取最小后的目标 | 解释 |
|---:|---:|---:|---:|---|
| $1.10$ | $2.20$ | $2.20$ | $2.20$ | 合理提高动作概率 |
| $1.20$ | $2.40$ | $2.40$ | $2.40$ | 到达收益边界 |
| $1.50$ | $3.00$ | $2.40$ | $2.40$ | 再提高也没有额外奖励 |
| $0.50$ | $1.00$ | $1.60$ | $1.00$ | 错误降低好动作，仍被惩罚 |

当正优势动作的概率已增加超过 $1+\epsilon$ 后，继续增加其概率无法提高代理目标。

### 6.2 优势为负：这是一个值得降低概率的动作

若：

$$
\hat{A}_t=-2<0
$$

| $r_t$ | $r_t\hat{A}_t$ | 裁剪项 | 取最小后的目标 | 解释 |
|---:|---:|---:|---:|---|
| $0.90$ | $-1.80$ | $-1.80$ | $-1.80$ | 合理降低坏动作概率 |
| $0.80$ | $-1.60$ | $-1.60$ | $-1.60$ | 到达改善边界 |
| $0.50$ | $-1.00$ | $-1.60$ | $-1.60$ | 再降低也没有额外奖励 |
| $1.50$ | $-3.00$ | $-2.40$ | $-3.00$ | 错误提高坏动作，仍被惩罚 |

### 6.3 最重要的理解

PPO 并不是把所有策略概率变化硬性裁剪掉，而是：

$$
\boxed{
\text{限制向“有利方向”改变过多所获得的额外收益，同时保留错误方向的惩罚。}
}
$$

因此 PPO 使用 $\min$，而不是仅使用裁剪后的概率比。

---

## 7. PPO 中的 Critic 与价值目标

PPO 通常同时训练 Critic：

$$
V_\phi(S_t)
$$

使用优势估计与旧价值构造回报目标：

$$
\hat{R}_t
=
\hat{A}_t+V_{\phi_{\text{old}}}(S_t)
$$

基础价值损失为：

$$
L_{\text{value}}
=
\frac{1}{2}
\mathbb{E}_t\left[
\left(V_\phi(S_t)-\hat{R}_t\right)^2
\right]
$$

Critic 越能准确估计状态价值，Actor 使用的优势信号通常越可靠。

---

## 8. GAE：PPO 中常用的优势估计方法

PPO 通常使用 GAE（Generalized Advantage Estimation，广义优势估计）构造 $\hat{A}_t$。

### 8.1 一步 TD 误差

$$
\delta_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

### 8.2 GAE 公式

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
$$

展开即：

$$
\hat{A}_t
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
$$

递推实现为：

$$
\hat{A}_t
=
\delta_t
+
\gamma\lambda(1-e_t)\hat{A}_{t+1}
$$

其中 $e_t$ 表示 episode 是否在该转移后结束；递推不得跨越 reset 边界。

### 8.3 $\lambda$ 的偏差—方差权衡

| $\lambda$ | 更接近 | 特点 |
|---:|---|---|
| $0$ | 一步 TD | 方差较小，但依赖价值函数，偏差可能较大 |
| 接近 $1$ | 蒙特卡洛回报 | 偏差较小，但方差较大 |
| 中间取值 | 折中 | PPO 中常用，例如 $0.95$ |

---

## 9. 策略熵与完整总损失

为避免策略过早退化为几乎确定性的动作，PPO 常加入熵奖励：

$$
\mathcal{H}\left(\pi_\theta(\cdot\mid S_t)\right)
=
-\sum_a\pi_\theta(a\mid S_t)\log\pi_\theta(a\mid S_t)
$$

需要最小化的完整损失通常写为：

$$
L_{\text{total}}
=
-L^{\mathrm{CLIP}}(\theta)
+
c_v L_{\text{value}}
-
c_e\mathcal{H}(\pi_\theta)
$$

其中：

| 符号 | 含义 |
|---|---|
| $c_v$ | 价值损失权重 |
| $c_e$ | 熵奖励权重 |
| $L^{\mathrm{CLIP}}$ | PPO 策略裁剪目标 |
| $L_{\text{value}}$ | Critic 价值损失 |

---

## 10. 为什么 PPO 可对同一批数据训练多个 epoch

PPO 的数据仍然来自当前旧策略，因此 PPO 是 on-policy 算法。与普通一次策略梯度更新不同，PPO：

1. 保存旧动作概率；
2. 每次训练时计算新旧策略比值；
3. 对过度有利的概率变化进行裁剪；
4. 因此可对一批 rollout 数据进行若干轮 minibatch 更新。

但是，PPO 并未将数据变成可无限复用的 off-policy 数据。若 update epochs 过多或学习率过大，新策略仍会远离旧策略。因此实践中常监控近似 KL 散度，并在偏移过大时提前终止本轮更新。

---

## 11. PPO 的完整训练流程

一次 PPO 迭代通常包含两个阶段。

### 11.1 Rollout 采集阶段

使用当前冻结的旧策略与环境交互若干步，保存：

$$
(S_t,A_t,R_{t+1},d_t,
\log\pi_{\theta_{\text{old}}}(A_t\mid S_t),
V_{\phi_{\text{old}}}(S_t))
$$

### 11.2 Update 更新阶段

1. 计算 TD 误差；
2. 使用 GAE 得到优势 $\hat{A}_t$；
3. 构造价值目标 $\hat{R}_t$；
4. 对优势进行标准化；
5. 将 rollout 数据打乱并切分 minibatch；
6. 对同一批数据训练多个 update epoch；
7. 每次重新计算新策略动作概率；
8. 计算概率比、裁剪策略损失、价值损失与熵；
9. 更新 Actor 与 Critic；
10. 丢弃该批 rollout，使用更新后的策略重新采集数据。

---

## 12. PPO-Clip 伪代码

```text
初始化 Actor 参数 θ 与 Critic 参数 φ

重复执行每一次迭代：

    固定旧策略 π_old = π_θ
    使用 π_old 与环境交互，采集 rollout：
        S_t, A_t, R_{t+1}, done_t,
        log π_old(A_t | S_t), V_old(S_t)

    计算 TD 误差：
        δ_t = R_{t+1} + γ V_old(S_{t+1}) - V_old(S_t)

    使用 GAE 计算优势：
        Â_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...

    构造价值目标：
        R̂_t = Â_t + V_old(S_t)

    标准化优势 Â_t

    重复 K 个 update epochs：
        打乱 rollout 数据并分成 minibatches

        对每个 minibatch：
            重新计算 log π_θ(A_t | S_t)
            r_t = exp(log π_θ - log π_old)

            surr1 = r_t Â_t
            surr2 = clip(r_t, 1-ε, 1+ε) Â_t

            L_policy = -mean(min(surr1, surr2))
            L_value = 1/2 mean((V_φ(S_t) - R̂_t)^2)
            H = mean(entropy(π_θ(· | S_t)))

            L = L_policy + c_v L_value - c_e H
            反向传播并更新参数
```

---

## 13. PyTorch 核心实现：离散动作 PPO-Clip

下面代码覆盖网络、GAE 与 PPO 多轮 minibatch 更新核心，可嵌入 `Gymnasium` 环境采样循环中使用。

```python
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """适用于离散动作空间的共享主干 Actor-Critic 网络。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor):
        features = self.backbone(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def act(self, states: torch.Tensor):
        logits, values = self(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    old_values: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    episode_ended: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_ended: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """
    terminated：任务真正终止，下一价值不可 bootstrap。
    episode_ended：terminated 或 truncated，GAE 递推不得跨过 reset 边界。
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(len(rewards))):
        bootstrap_mask = 1.0 - terminated[t]
        continuation_mask = 1.0 - episode_ended[t]

        delta = (
            rewards[t]
            + gamma * bootstrap_mask * next_values[t]
            - values[t]
        )
        gae = delta + gamma * gae_lambda * continuation_mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    batch: RolloutBatch,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    max_grad_norm: float = 0.5,
):
    with torch.no_grad():
        advantages, returns = compute_gae(
            batch.rewards,
            batch.old_values,
            batch.next_values,
            batch.terminated,
            batch.episode_ended,
            gamma,
            gae_lambda,
        )
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

    sample_count = batch.states.size(0)
    metrics = {}

    for _ in range(update_epochs):
        permutation = torch.randperm(sample_count, device=batch.states.device)

        for start in range(0, sample_count, minibatch_size):
            idx = permutation[start:start + minibatch_size]

            new_log_probs, entropy, new_values = model.evaluate_actions(
                batch.states[idx], batch.actions[idx]
            )

            old_log_probs = batch.old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_1 = ratio * mb_advantages
            surrogate_2 = torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            ) * mb_advantages

            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = 0.5 * (mb_returns - new_values).pow(2).mean()
            entropy_bonus = entropy.mean()

            total_loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy_bonus
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_epsilon
                ).float().mean()

            metrics = {
                "total_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_bonus.item(),
                "approx_kl": approx_kl.item(),
                "clip_fraction": clip_fraction.item(),
            }

    return metrics
```

---

## 14. 代码与公式对应关系

### 14.1 概率比

```python
ratio = torch.exp(new_log_probs - old_log_probs)
```

对应：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(A_t\mid S_t)
-
\log\pi_{\theta_{\text{old}}}(A_t\mid S_t)
\right)
$$

### 14.2 裁剪代理目标

```python
surrogate_1 = ratio * mb_advantages
surrogate_2 = torch.clamp(
    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
) * mb_advantages
policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
```

对应 PPO-Clip 损失：

$$
L_{\text{policy}}
=
-\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t
\right)
\right]
$$

### 14.3 GAE 与价值目标

```python
advantages, returns = compute_gae(...)
```

对应：

$$
\hat{A}_t
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

以及：

$$
\hat{R}_t
=
\hat{A}_t+V_{\phi_{\text{old}}}(S_t)
$$

---

## 15. `terminated` 与 `truncated` 的区别

在 Gymnasium 环境中：

| 类型 | 含义 | 是否允许价值 bootstrap | 是否允许 GAE 跨 reset 递推 |
|---|---|---|---|
| `terminated=True` | 环境定义的真正终止 | 否 | 否 |
| `truncated=True` | 时间上限等外部截断 | 通常允许 | 否 |

因此代码中通常需要两个 mask：

- `bootstrap_mask`：控制下一状态价值是否加入 TD 目标；
- `continuation_mask`：控制 GAE 是否继续向上一时刻递推。

---

## 16. 重点监控指标

| 指标 | 含义 | 诊断意义 |
|---|---|---|
| Episode Return | 每回合累计奖励 | 最直接的任务性能指标 |
| Policy Loss | 策略更新目标 | 剧烈异常可能表示策略更新不稳 |
| Value Loss | Critic 拟合误差 | 过大说明价值估计不可靠 |
| Entropy | 动作分布随机程度 | 过快降到近零说明探索不足 |
| Approx KL | 新旧策略偏移近似量 | 过大说明更新过猛 |
| Clip Fraction | 被裁剪的样本比例 | 过高表示大量更新触碰裁剪边界 |
| Explained Variance | 价值函数解释能力 | 接近或低于零表示 Critic 质量较差 |

---

## 17. 常见超参数起点

| 超参数 | 含义 | 常见起点 |
|---|---|---:|
| $\gamma$ | 折扣因子 | $0.99$ |
| $\lambda$ | GAE 参数 | $0.95$ |
| $\epsilon$ | 裁剪范围 | $0.2$ |
| learning rate | 学习率 | $3\times10^{-4}$ |
| update epochs | 每批数据优化轮数 | $4$ 至 $10$ |
| minibatch size | 小批量大小 | $64$ 或 $128$ |
| $c_v$ | 价值损失权重 | $0.5$ |
| $c_e$ | 熵系数 | $0$ 至 $0.01$ |
| max grad norm | 梯度裁剪阈值 | $0.5$ |

这些数值适合作为入门实验起点，而不是所有任务上的固定最佳配置。

---

## 18. 常见错误

### 18.1 未保存旧策略动作概率

没有：

$$
\log\pi_{\theta_{\text{old}}}(A_t\mid S_t)
$$

就无法计算 PPO 的核心概率比。

### 18.2 采样过程中修改策略

一批 rollout 应由固定旧策略产生。标准流程是先完成采样，再进行多个 epoch 的更新。

### 18.3 同一批数据更新太久

PPO 仍是 on-policy 方法。同一批数据被复用太久会使新策略逐渐远离生成这些数据的旧策略。

### 18.4 把裁剪理解为硬约束

PPO-Clip 没有对所有状态的 KL 散度施加严格约束。它是在采样动作的代理目标中削弱过度获利更新的激励；因此仍应监控 KL 偏移。

### 18.5 GAE 跨越 reset 边界

回合重置后，新 episode 的价值信息不应继续回传给前一 episode；时间截断尤其需要正确处理递推边界。

---

## 19. PPO 与前序算法的对比

| 对比项 | REINFORCE | Actor-Critic | PPO-Clip |
|---|---|---|---|
| 策略信号 | 蒙特卡洛回报 | 优势 / TD 误差 | 优势 + 裁剪概率比 |
| Critic | 原始版本无 | 有 | 通常有 |
| 单批数据使用 | 通常一次 | 需谨慎 | 允许若干轮受控更新 |
| 更新稳定性 | 较低 | 较好 | 通常更好 |
| 核心难点 | 方差大 | 策略仍可能更新过猛 | 裁剪、GAE 与训练细节 |

---

## 20. PPO 与 TRPO 的关系

TRPO 通过 KL 散度约束构造信赖域：

$$
\mathbb{E}_t\left[
D_{\mathrm{KL}}\left(
\pi_{\theta_{\text{old}}}(\cdot\mid S_t)
\;\|\;
\pi_\theta(\cdot\mid S_t)
\right)
\right]
\le \delta
$$

TRPO 的实现涉及较复杂的约束优化和二阶近似。PPO 则使用更简单的一阶优化流程，通过裁剪概率比构造保守代理目标，近似实现“不要一次离旧策略太远”的目的。

---

## 21. 必须掌握的核心公式

### 概率比

$$
r_t(\theta)
=
\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{\text{old}}}(A_t\mid S_t)}
$$

### PPO-Clip 目标

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t
\right)
\right]
$$

### TD 误差

$$
\delta_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

### GAE

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

### 价值损失

$$
L_{\text{value}}
=
\frac{1}{2}\mathbb{E}_t\left[
\left(V_\phi(S_t)-\hat{R}_t\right)^2
\right]
$$

### 总损失

$$
L_{\text{total}}
=
-L^{\mathrm{CLIP}}(\theta)
+c_vL_{\text{value}}
-c_e\mathcal{H}(\pi_\theta)
$$

---

## 22. 总结

PPO 的核心过程是：

1. 使用旧策略收集一批 on-policy 数据；
2. 使用 Critic 与 GAE 得到动作优势；
3. 计算新旧策略对同一动作的概率比；
4. 通过裁剪目标限制策略向有利方向改变得过多；
5. 同时优化 Actor、Critic，并用熵奖励保持探索。

最核心的一句话是：

$$
\boxed{
\text{Actor-Critic 判断策略该往哪里改，PPO-Clip 限制策略一次不要改得过头。}
}
$$

---

## 参考文献

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). *Trust Region Policy Optimization*. arXiv:1502.05477.
3. Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438.
4. OpenAI. *Spinning Up: Proximal Policy Optimization*.
