# Actor-Critic 算法详解：从 REINFORCE 到优势函数与 TD 更新

> 本笔记以**标准的一步 on-policy Advantage Actor-Critic** 为主线。Actor-Critic 不是单个固定算法，而是一类同时学习策略（Actor）与价值估计（Critic）的策略梯度方法。

---

## 1. 为什么从 REINFORCE 发展到 Actor-Critic

在 REINFORCE 中，策略通过一条完整轨迹的蒙特卡洛回报更新：

$$
\nabla_\theta J(\theta)
\approx
\sum_{t=0}^{T-1}
G_t
\nabla_\theta \log \pi_\theta(A_t \mid S_t)
$$

其中：

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+\cdots
$$

对应的策略损失为：

$$
L_{\text{REINFORCE}}
=
-
\sum_{t=0}^{T-1}
G_t
\log \pi_\theta(A_t \mid S_t)
$$

REINFORCE 存在两个核心问题：

1. **必须等待整局结束**：在得到完整的 $G_t$ 前，不能更新当前动作。
2. **梯度方差大**：同一个动作之后的完整未来回报可能受到很多后续随机动作和环境随机性的影响，导致信用分配不准确。

Actor-Critic 的出发点是：

> 不再仅仅等待整局结束后用真实回报评价动作，而是另外训练一个 Critic，随时估计当前策略下状态或动作的价值，并用这个评价信号指导 Actor 更新。

---

## 2. Actor-Critic 的整体结构

Actor-Critic 包含两个角色：

| 模块 | 学习对象 | 功能 |
|---|---|---|
| Actor | 策略 $\pi_\theta(a \mid s)$ | 根据状态决定采取什么动作 |
| Critic | 价值函数 $V_\phi(s)$ 或 $Q_\phi(s,a)$ | 评价 Actor 当前动作是否优于平均水平 |

其中：

- $\theta$ 是 Actor 的参数；
- $\phi$ 是 Critic 的参数；
- Actor 负责“行动”；
- Critic 负责“打分”。

最常见的结构是 Critic 学习状态价值函数：

$$
V_\phi(s)
\approx
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t \mid S_t=s
\right]
$$

也就是说，Critic 回答的问题是：

> 在状态 $s$ 下，按照当前策略继续行动，平均可以获得多少累计回报？

Actor 则学习：

$$
\pi_\theta(a \mid s)
$$

也就是说，Actor 回答的问题是：

> 在状态 $s$ 下，应当以多大概率采取动作 $a$？

---

## 3. 从策略梯度到优势 Actor-Critic

策略梯度定理给出：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{S_t,A_t}
\left[
Q^\pi(S_t,A_t)
\nabla_\theta
\log
\pi_\theta(A_t \mid S_t)
\right]
$$

这里 $Q^\pi(S_t,A_t)$ 表示：在状态 $S_t$ 执行动作 $A_t$ 后，再按照当前策略行动，期望可以获得多少未来回报。

但是直接使用 $Q^\pi(S_t,A_t)$ 有一个问题：一个状态本身可能就很容易获得高奖励。即使某个动作很普通，它的 $Q$ 值也可能较高。

因此，引入状态价值作为基线：

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
$$

其中 $A^\pi(s,a)$ 称为**优势函数**。

它回答的问题不再是：

> 这个动作最后能得到多少奖励？

而是：

> 这个动作比当前状态下的平均动作表现好多少？

将策略梯度改写为：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{S_t,A_t}
\left[
A^\pi(S_t,A_t)
\nabla_\theta
\log
\pi_\theta(A_t \mid S_t)
\right]
$$

因此：

- 若 $A^\pi(S_t,A_t)>0$，当前动作优于平均水平，应提高其概率；
- 若 $A^\pi(S_t,A_t)<0$，当前动作差于平均水平，应降低其概率；
- 若 $A^\pi(S_t,A_t)\approx 0$，当前动作与平均水平接近，策略无需明显调整。

---

## 4. Critic 如何估计优势：TD 误差

### 4.1 价值函数的 Bellman 关系

对当前策略 $\pi$，状态价值函数满足：

$$
V^\pi(S_t)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid S_t
\right]
$$

对于一次实际采样，可以构造一步 TD 目标：

$$
y_t
=
R_{t+1}
+
\gamma (1-d_t)
V_\phi(S_{t+1})
$$

其中：

- $d_t=1$ 表示该转移后回合真正终止；
- $d_t=0$ 表示回合仍可继续；
- $(1-d_t)$ 用于终止状态不再 bootstrap。

TD 误差定义为：

$$
\delta_t
=
y_t - V_\phi(S_t)
$$

即：

$$
\delta_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

### 4.2 为什么 TD 误差可以近似优势函数

如果 Critic 已经能够准确估计真实状态价值，即 $V_\phi \approx V^\pi$，则在给定 $(S_t=s,A_t=a)$ 的条件下：

$$
\mathbb{E}
\left[
\delta_t
\mid
S_t=s,A_t=a
\right]
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s,A_t=a
\right]
-
V^\pi(s)
$$

而：

$$
\mathbb{E}
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s,A_t=a
\right]
=
Q^\pi(s,a)
$$

因此：

$$
\mathbb{E}
\left[
\delta_t
\mid
S_t=s,A_t=a
\right]
=
Q^\pi(s,a)-V^\pi(s)
=
A^\pi(s,a)
$$

所以，一步 TD 误差 $\delta_t$ 可以作为优势函数 $A^\pi(S_t,A_t)$ 的采样估计。

这就是标准 Actor-Critic 最关键的一步：

$$
\boxed{
\text{Critic 的 TD 误差 }\delta_t
\text{ 作为 Actor 更新所需的优势信号}
}
$$

---

## 5. Actor 与 Critic 各自如何更新

### 5.1 Actor 的更新

Actor 希望提高优势为正的动作概率，降低优势为负的动作概率。

使用 TD 误差近似优势后，策略梯度估计为：

$$
\widehat{\nabla_\theta J(\theta)}
=
\delta_t
\nabla_\theta
\log
\pi_\theta(A_t \mid S_t)
$$

采用梯度上升时：

$$
\theta
\leftarrow
\theta
+
\alpha_\theta
\delta_t
\nabla_\theta
\log
\pi_\theta(A_t \mid S_t)
$$

深度学习框架默认最小化损失，因此 Actor 损失写为：

$$
L_{\text{actor}}
=
-
\operatorname{stopgrad}(\delta_t)
\log
\pi_\theta(A_t \mid S_t)
$$

这里的 $\operatorname{stopgrad}(\delta_t)$ 表示更新 Actor 时将优势视作固定评分，不让 Actor 损失反向传播去改变 Critic。

### 5.2 Critic 的更新

Critic 的任务是让自己的估计 $V_\phi(S_t)$ 接近 TD 目标：

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
$$

通常把 TD 目标停止梯度后，使用均方误差：

$$
L_{\text{critic}}
=
\frac{1}{2}
\left[
\operatorname{stopgrad}(y_t)
-
V_\phi(S_t)
\right]^2
$$

通过最小化该损失，Critic 学会更准确地预测当前状态的价值。

### 5.3 熵奖励：防止策略过早确定化

实际训练中常加入熵正则项：

$$
\mathcal{H}
\left(
\pi_\theta(\cdot \mid S_t)
\right)
=
-
\sum_a
\pi_\theta(a \mid S_t)
\log
\pi_\theta(a \mid S_t)
$$

策略熵越大，说明动作分布越分散，探索性越强。

综合 Actor 损失可写为：

$$
L_{\text{actor-total}}
=
-
\delta_t
\log
\pi_\theta(A_t \mid S_t)
-
\beta
\mathcal{H}
\left(
\pi_\theta(\cdot \mid S_t)
\right)
$$

其中 $\beta>0$ 为熵系数。

### 5.4 总损失

若 Actor 和 Critic 使用同一个共享网络主体，常见总损失为：

$$
L
=
L_{\text{actor}}
+
c_v L_{\text{critic}}
-
\beta \mathcal{H}(\pi_\theta)
$$

展开后：

$$
L
=
-
\operatorname{stopgrad}(\delta_t)
\log
\pi_\theta(A_t \mid S_t)
+
\frac{c_v}{2}
\left[
\operatorname{stopgrad}(y_t)-V_\phi(S_t)
\right]^2
-
\beta
\mathcal{H}
\left(
\pi_\theta(\cdot \mid S_t)
\right)
$$

---

## 6. 标准一步 Actor-Critic 的完整流程

每个时间步执行以下过程：

1. 当前状态为 $S_t$。
2. Actor 输出动作分布 $\pi_\theta(\cdot \mid S_t)$。
3. 从分布中采样动作 $A_t \sim \pi_\theta(\cdot \mid S_t)$。
4. 在环境中执行动作，得到 $R_{t+1}$、$S_{t+1}$ 与 $d_t$。
5. Critic 计算 $V_\phi(S_t)$ 与 $V_\phi(S_{t+1})$。
6. 构造 TD 目标：

   $$
   y_t
   =
   R_{t+1}
   +
   \gamma(1-d_t)V_\phi(S_{t+1})
   $$

7. 计算 TD 误差：

   $$
   \delta_t
   =
   y_t-V_\phi(S_t)
   $$

8. 使用 $\delta_t$ 更新 Actor：

   $$
   L_{\text{actor}}
   =
   -
   \delta_t
   \log
   \pi_\theta(A_t \mid S_t)
   $$

9. 使用 TD 目标更新 Critic：

   $$
   L_{\text{critic}}
   =
   \frac{1}{2}
   \left(
   y_t-V_\phi(S_t)
   \right)^2
   $$

10. 令 $S_t \leftarrow S_{t+1}$，继续交互。

与 REINFORCE 不同，标准一步 Actor-Critic 不必等待完整回合结束就可以更新。

---

## 7. 数值例子：TD 误差如何同时指导 Actor 和 Critic

设某一时刻：

$$
V_\phi(S_t)=4.0
$$

执行动作 $A_t$ 后，获得奖励：

$$
R_{t+1}=2.0
$$

下一状态的价值估计为：

$$
V_\phi(S_{t+1})=5.0
$$

设折扣因子为 $\gamma=0.9$，且当前转移不是终止状态，即 $d_t=0$。

### 7.1 计算 TD 目标

$$
y_t
=
2.0+0.9\times5.0
=
6.5
$$

### 7.2 计算 TD 误差

$$
\delta_t
=
6.5-4.0
=
2.5
$$

因为 $\delta_t>0$，说明此次动作带来的结果比 Critic 原先预期的更好。因此：

- Actor 提高 $\pi_\theta(A_t \mid S_t)$；
- Critic 将 $V_\phi(S_t)$ 从 $4.0$ 向 $6.5$ 调高。

### 7.3 若 TD 误差为负

若实际得到：

$$
R_{t+1}=0,\qquad V_\phi(S_{t+1})=2.0
$$

则：

$$
y_t
=
0+0.9\times2.0
=
1.8
$$

$$
\delta_t
=
1.8-4.0
=
-2.2
$$

此时：

- Actor 降低该动作以后被选中的概率；
- Critic 将当前状态价值预测向下修正。

---

## 8. 一步 Actor-Critic 伪代码

```text
输入：
    策略网络 πθ(a|s)
    状态价值网络 Vφ(s)
    折扣因子 γ
    Actor 学习率 αθ
    Critic 学习率 αφ

初始化 θ, φ

对每个 episode：
    初始化状态 S

    while episode 未结束：

        根据 πθ(·|S) 采样动作 A
        执行动作 A，得到奖励 R、下一状态 S' 和终止标记 d

        计算当前状态价值：
            V = Vφ(S)

        计算 TD 目标：
            y = R + γ(1-d)Vφ(S')

        计算 TD 误差：
            δ = y - V

        更新 Actor：
            θ ← θ + αθ δ ∇θ log πθ(A|S)

        更新 Critic：
            最小化 1/2 (y - Vφ(S))²

        S ← S'
```

---

## 9. PyTorch 实现：离散动作的一步 Actor-Critic

下面给出一个可用于 `Gymnasium CartPole-v1` 的基础实现。为了便于理解，Actor 和 Critic 使用共享特征提取层，并分别输出动作 logits 与状态价值。

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """共享主干的一步 Actor-Critic 网络，适用于离散动作空间。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value


def train_actor_critic(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    gamma: float = 0.99,
    learning_rate: float = 3e-4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    torch.manual_seed(seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    recent_returns = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            logits, value = model(state_tensor)
            distribution = Categorical(logits=logits)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            terminated_tensor = torch.tensor([float(terminated)], dtype=torch.float32)

            with torch.no_grad():
                next_state_tensor = torch.tensor(
                    next_state, dtype=torch.float32
                ).unsqueeze(0)
                _, next_value = model(next_state_tensor)

                # 真正终止状态不能 bootstrap；时间截断通常仍可 bootstrap。
                td_target = (
                    reward_tensor
                    + gamma * (1.0 - terminated_tensor) * next_value
                )

            advantage = td_target - value

            actor_loss = -(advantage.detach() * log_prob).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            entropy_bonus = entropy.mean()

            loss = (
                actor_loss
                + value_coef * critic_loss
                - entropy_coef * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            state = next_state
            episode_return += reward

        recent_returns.append(episode_return)
        if len(recent_returns) > 20:
            recent_returns.pop(0)

        if episode % 20 == 0:
            mean_return = sum(recent_returns) / len(recent_returns)
            print(
                f"Episode {episode:4d} | "
                f"Return {episode_return:7.2f} | "
                f"Mean20 {mean_return:7.2f}"
            )

    env.close()
    return model


if __name__ == "__main__":
    trained_model = train_actor_critic()
```

---

## 10. 代码与公式逐行对应

### 10.1 Actor 输出动作分布

```python
logits, value = model(state_tensor)
distribution = Categorical(logits=logits)
action = distribution.sample()
log_prob = distribution.log_prob(action)
```

对应：

$$
A_t \sim \pi_\theta(\cdot \mid S_t)
$$

以及：

$$
\log \pi_\theta(A_t \mid S_t)
$$

### 10.2 Critic 构造 TD 目标

```python
td_target = reward_tensor + gamma * (1.0 - terminated_tensor) * next_value
```

对应：

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
$$

### 10.3 TD 误差作为优势估计

```python
advantage = td_target - value
```

对应：

$$
\delta_t
=
y_t-V_\phi(S_t)
$$

### 10.4 Actor 损失

```python
actor_loss = -(advantage.detach() * log_prob).mean()
```

对应：

$$
L_{\text{actor}}
=
-
\operatorname{stopgrad}(\delta_t)
\log
\pi_\theta(A_t \mid S_t)
$$

`detach()` 非常关键：Actor 应把 Critic 给出的优势当作评分，而不是通过 Actor 损失直接反向修改 Critic 的预测结构。

### 10.5 Critic 损失

```python
critic_loss = 0.5 * advantage.pow(2).mean()
```

对应：

$$
L_{\text{critic}}
=
\frac{1}{2}
\left[
y_t-V_\phi(S_t)
\right]^2
$$

### 10.6 熵奖励

```python
entropy_bonus = entropy.mean()
loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus
```

对应：

$$
L
=
L_{\text{actor}}
+
c_v L_{\text{critic}}
-
\beta \mathcal{H}(\pi_\theta)
$$

最小化损失时减去熵，等价于鼓励策略保持一定探索性。

---

## 11. 为什么 `terminated` 与 `truncated` 需要区分

在 Gymnasium 中：

- `terminated=True`：任务按照环境定义真正结束，例如小车倒下；
- `truncated=True`：因为时间上限等外部限制而截断。

对于真正终止状态，未来价值为零：

$$
y_t=R_{t+1}
$$

但对于时间截断，状态本身不一定是终止状态，其后仍可能具有价值。因此，通常仍应 bootstrap：

$$
y_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
$$

若简单把两者都视为未来价值为零，会给 Critic 引入不必要的偏差。

---

## 12. Actor-Critic 与 REINFORCE 的本质区别

| 项目 | REINFORCE | Actor-Critic |
|---|---|---|
| 策略网络 | 有 | 有 |
| 价值网络 | 原始版本无 | 有 |
| 动作评价信号 | 完整回报 $G_t$ | Critic 给出的优势或 TD 误差 |
| 更新时机 | 通常回合结束后 | 可逐步或分段更新 |
| 方差 | 较高 | 通常更低 |
| 偏差 | 蒙特卡洛回报通常低偏差 | bootstrap 引入一定偏差 |
| 样本效率 | 较低 | 通常更高 |
| 训练复杂度 | 简单 | 更复杂，需要平衡 Actor 与 Critic |

关键权衡是：

$$
\text{REINFORCE：低偏差，高方差}
$$

$$
\text{Actor-Critic：引入一定偏差，换取更低方差和更及时的更新}
$$

---

## 13. Actor-Critic 与 DQN 的区别

| 项目 | DQN | Actor-Critic |
|---|---|---|
| 核心学习对象 | 动作价值 $Q(s,a)$ | 策略 $\pi(a|s)$ 与价值估计 |
| 策略表示 | 通常由 $\arg\max_a Q(s,a)$ 隐式产生 | 显式策略网络 |
| 动作空间 | 主要适合离散动作 | 可扩展到离散与连续动作 |
| 随机策略 | 通常额外加入探索机制 | 策略本身可为随机分布 |
| 训练方式 | TD 值学习 | 策略梯度 + TD 评价 |

---

## 14. Actor-Critic 的主要变体

### 14.1 A2C：Advantage Actor-Critic

A2C 通常使用多个并行环境同步收集一段轨迹，再统一更新，常使用 $n$ 步回报或 GAE 估计优势。

### 14.2 A3C：Asynchronous Advantage Actor-Critic

A3C 使用多个异步 worker 并行与不同环境副本交互，各自异步更新共享参数。并行且较不相关的数据流可以改善深度强化学习的训练表现。

### 14.3 PPO：受约束的 Actor-Critic 更新

PPO 通常包含 Actor 和 Critic，并通过裁剪概率比限制一次策略更新幅度：

$$
r_t(\theta)
=
\frac{
\pi_\theta(A_t \mid S_t)
}{
\pi_{\theta_{\text{old}}}(A_t \mid S_t)
}
$$

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}
\left(
r_t(\theta),1-\epsilon,1+\epsilon
\right)
\hat{A}_t
\right)
\right]
$$

### 14.4 DDPG、TD3 与 SAC：连续动作 Actor-Critic

在连续动作控制中：

- DDPG、TD3 使用确定性 Actor 和 $Q$ Critic；
- SAC 使用随机策略与熵正则，强调探索与鲁棒性。

它们都属于广义的 Actor-Critic 框架。

---

## 15. 从一步 TD 到多步回报与 GAE

一步 TD 更新响应快，但可能偏差较大：

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

多步目标在真实奖励与 bootstrap 之间取得折中。例如 $n$ 步目标为：

$$
G_t^{(n)}
=
R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n}
+
\gamma^n V(S_{t+n})
$$

优势估计可写为：

$$
\hat{A}_t^{(n)}
=
G_t^{(n)}-V(S_t)
$$

现代 PPO 等算法常使用广义优势估计 GAE：

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$

其中：

- $\lambda$ 较小：更接近一步 TD，方差小但偏差更大；
- $\lambda$ 较大：更接近蒙特卡洛回报，偏差小但方差更大。

---

## 16. 常见错误与训练建议

### 16.1 Actor 损失中没有停止优势梯度

错误思路：

```python
actor_loss = -(advantage * log_prob).mean()
```

当 `advantage` 中包含 Critic 的可导输出时，Actor 损失会额外影响 Critic 路径，导致优化目标混乱。

建议写法：

```python
actor_loss = -(advantage.detach() * log_prob).mean()
```

### 16.2 终止状态仍进行 bootstrap

若任务真正结束，仍然使用 $V(S_{t+1})$，会使 Critic 凭空估计终止后的收益。

正确做法：

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)V(S_{t+1})
$$

### 16.3 熵系数过大或过小

- 熵系数太小：策略可能很早退化为几乎确定的动作，探索不足；
- 熵系数太大：策略迟迟不能集中到优良动作上。

### 16.4 Critic 学得太差

Actor 完全依赖 Critic 提供更新方向。若 Critic 的价值估计误差很大，Actor 可能被错误指导。

可采取的措施包括：

- 调整价值损失权重；
- 使用合适的学习率；
- 使用 $n$ 步回报或 GAE；
- 对优势进行标准化；
- 使用梯度裁剪；
- 监控 value loss 与 explained variance。

---

## 17. 必须掌握的核心公式

### 状态价值函数

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t \mid S_t=s
\right]
$$

### 优势函数

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
$$

### 一步 TD 目标

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
$$

### TD 误差 / 一步优势估计

$$
\delta_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

### Actor 损失

$$
L_{\text{actor}}
=
-
\operatorname{stopgrad}(\delta_t)
\log
\pi_\theta(A_t \mid S_t)
$$

### Critic 损失

$$
L_{\text{critic}}
=
\frac{1}{2}
\left[
\operatorname{stopgrad}(y_t)-V_\phi(S_t)
\right]^2
$$

### 带熵奖励的联合损失

$$
L
=
L_{\text{actor}}
+
c_vL_{\text{critic}}
-
\beta
\mathcal{H}
\left(
\pi_\theta(\cdot \mid S_t)
\right)
$$

---

## 18. 一句话总结

Actor-Critic 的核心逻辑是：

$$
\boxed{
\text{Actor 负责选择动作，Critic 估计该动作是否优于预期，TD 误差同时推动策略改进与价值修正。}
}
$$

相比 REINFORCE，Actor-Critic 用价值网络的 bootstrap 评价替代单纯等待整局回报，从而能够更及时、更低方差地更新策略；代价是引入了 Critic 估计误差和一定的偏差。

---

## 参考文献

1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 13: Policy Gradient Methods.
2. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS.
3. Konda, V. R., & Tsitsiklis, J. N. (1999). *Actor-Critic Algorithms*. NeurIPS.
4. Mnih, V., et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML.
