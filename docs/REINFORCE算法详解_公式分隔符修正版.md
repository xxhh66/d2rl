# REINFORCE 算法详解

## 1. REINFORCE 是什么

**REINFORCE** 是最经典的策略梯度算法之一。它不先学习动作价值函数再选择动作，而是直接学习参数化策略：
$$
\pi_\theta(a \mid s)
$$

其核心思想是：

> 按照当前策略完成一次交互轨迹；若轨迹中某些动作之后获得了较高回报，就提高这些动作在相应状态下再次被选择的概率；若回报较低，则降低其概率。

REINFORCE 具有以下特征：

- **策略梯度方法**：直接优化策略网络 $\pi_\theta(a \mid s)$；
- **蒙特卡洛方法**：依赖完整轨迹结束后的实际累计回报；
- **On-policy 方法**：更新所使用的轨迹由当前策略采样产生；
- **随机策略方法**：训练阶段从策略分布中采样动作，以维持探索。

---

## 2. 强化学习交互过程与优化目标

一条回合轨迹可以表示为：

$$
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots,S_{T-1},A_{T-1},R_T,S_T)
$$

其中：

- $S_t$：时刻 $t$ 的状态；
- $A_t$：策略选择的动作；
- $R_{t+1}$：执行动作 $A_t$ 后获得的奖励；
- $\gamma \in [0,1]$：折扣因子。

从时刻 $t$ 开始的折扣回报为：

$$
G_t
=
R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T
$$

也可以递归计算：

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

REINFORCE 的目标是最大化初始时刻的期望累计回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]
$$

---

## 3. REINFORCE 与 DQN 的区别

### 3.1 DQN：学习动作价值函数

DQN 学习：

$$
Q(s,a)
$$

决策时选择价值最高的动作：

$$
a^*=\arg\max_a Q(s,a)
$$

即：先估计各个动作的价值，再选择最优动作。

### 3.2 REINFORCE：直接学习动作分布

REINFORCE 直接学习：

$$
\pi_\theta(a \mid s)
$$

例如在某状态 $s$ 下，策略可能输出：

$$
\pi_\theta(\text{左}\mid s)=0.7,\qquad
\pi_\theta(\text{右}\mid s)=0.3
$$

训练阶段，智能体按照这个概率分布采样动作，而不是始终选概率最大的动作。

| 方法 | 学习对象 | 动作生成方式 | 典型特点 |
|---|---|---|---|
| DQN | $Q(s,a)$ | 通常选择最大价值动作并加入探索 | 适合离散动作 |
| REINFORCE | $\pi_\theta(a\mid s)$ | 从策略分布采样 | 原理简单，但方差较大 |
| Actor-Critic | 策略与价值函数 | Actor 采样，Critic 评价 | 降低方差 |
| PPO | 受约束的策略更新 | 使用旧策略数据稳定优化 | 训练更稳定 |

---

## 4. REINFORCE 的核心公式

策略梯度的一般形式为：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\gamma^tG_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right]
$$

REINFORCE 不精确计算上述期望，而是使用实际采样得到的轨迹近似梯度：

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_{t=0}^{T-1}
\gamma^tG_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

然后执行梯度上升：

$$
\theta
\leftarrow
\theta+
\alpha\widehat{\nabla_\theta J(\theta)}
$$

其中 $\alpha$ 为学习率。

实际编程中常将折扣已经包含在 $G_t$ 内，并使用如下估计：

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_{t=0}^{T-1}
G_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

---

## 5. 为什么使用对数概率 $\log\pi_\theta(A_t\mid S_t)$

一条轨迹中策略相关的概率部分是多个动作概率的乘积：

$$
\pi_\theta(A_0\mid S_0)
\pi_\theta(A_1\mid S_1)
\cdots
\pi_\theta(A_{T-1}\mid S_{T-1})
$$

对数能够把乘积变为求和：

$$
\log\left(
\prod_{t=0}^{T-1}\pi_\theta(A_t\mid S_t)
\right)
=
\sum_{t=0}^{T-1}\log\pi_\theta(A_t\mid S_t)
$$

并且满足对数求导技巧：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
$$

由于环境状态转移概率不依赖策略参数 $\theta$，轨迹对数概率的梯度最终只剩策略项：

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

因此，算法只需要知道：当前策略选中了什么动作，以及该动作之后获得了多少回报，不需要对环境求导。

---

## 6. 为什么要乘以回报 $G_t$

项

$$
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

表示提高动作 $A_t$ 在状态 $S_t$ 下被选择概率的参数方向。

但已采样动作不一定是好动作，因此要利用回报 $G_t$ 对其加权：

$$
G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

| $G_t$ 的情况 | 动作评价 | 策略调整效果 |
|---|---|---|
| $G_t$ 很大且为正 | 动作之后的结果较好 | 显著提高该动作概率 |
| $G_t$ 较小但为正 | 动作有收益但不突出 | 轻微提高该动作概率 |
| $G_t=0$ | 动作没有带来可见收益 | 基本不强化 |
| $G_t<0$ | 动作之后结果较差 | 降低该动作概率 |

因此可以将 REINFORCE 理解为：

$$
\text{策略更新}
=
\text{动作概率的可调整方向}
\times
\text{动作之后的实际收益}
$$

---

## 7. 为什么使用 $G_t$ 而不是所有时刻都使用 $G_0$

考虑一条三步轨迹：

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,S_3
$$

动作 $A_0$ 发生在所有奖励之前，因此它可能影响：

$$
R_1,\ R_2,\ R_3
$$

对应回报为：

$$
G_0=R_1+\gamma R_2+\gamma^2R_3
$$

动作 $A_2$ 发生在最后一步，它不可能影响已发生的 $R_1$ 和 $R_2$，只能影响：

$$
R_3
$$

因此：

$$
G_2=R_3
$$

这体现了**因果性原则**：当前动作只能根据它之后的奖励进行评价，不能把过去已经发生的奖励错误地归因给未来动作。

---

## 8. 具体示例：迷宫中的左右选择

假设机器人位于岔路口，动作只有向左或向右。初始策略为：

$$
\pi_\theta(\text{左}\mid s)=0.5,\qquad
\pi_\theta(\text{右}\mid s)=0.5
$$

### 第一次尝试

机器人采样到“左”，最终到达终点，获得：

$$
G_0=10
$$

由于回报较高，更新会提高“左”的概率，例如：

$$
\pi_\theta(\text{左}\mid s)=0.6,\qquad
\pi_\theta(\text{右}\mid s)=0.4
$$

### 第二次尝试

机器人采样到“右”，最终落入陷阱，获得：

$$
G_0=-10
$$

该次更新会降低“右”的概率，例如：

$$
\pi_\theta(\text{左}\mid s)=0.75,\qquad
\pi_\theta(\text{右}\mid s)=0.25
$$

经过大量轨迹采样后，策略逐渐偏向于能带来较高长期回报的动作。

---

## 9. 完整训练流程

### 步骤 1：初始化策略网络

建立带参数 $\theta$ 的策略网络：

$$
\pi_\theta(a\mid s)
$$

对于离散动作，网络通常先输出 logits，再通过 Softmax 转换为动作概率：

$$
\pi_\theta(a\mid s)=\operatorname{Softmax}(f_\theta(s))
$$

### 步骤 2：按照当前策略采样完整轨迹

在每个时刻 $t$：

1. 输入状态 $S_t$；
2. 得到动作概率分布 $\pi_\theta(\cdot\mid S_t)$；
3. 从分布中采样动作 $A_t$；
4. 执行动作，获得奖励 $R_{t+1}$ 和新状态 $S_{t+1}$；
5. 保存动作对数概率 $\log\pi_\theta(A_t\mid S_t)$ 与奖励；
6. 直到回合终止。

### 步骤 3：从后向前计算折扣回报

假设一局的奖励为：

$$
R_1=1,\qquad R_2=0,\qquad R_3=10
$$

令：

$$
\gamma=0.9
$$

则：

$$
G_2=R_3=10
$$

$$
G_1=R_2+\gamma G_2=0+0.9\times10=9
$$

$$
G_0=R_1+\gamma G_1=1+0.9\times9=9.1
$$

| 时刻 | 即时奖励 | 折扣回报 |
|---|---:|---:|
| $t=0$ | $R_1=1$ | $G_0=9.1$ |
| $t=1$ | $R_2=0$ | $G_1=9$ |
| $t=2$ | $R_3=10$ | $G_2=10$ |

### 步骤 4：构造策略损失函数

理论上，目标是最大化期望回报，因此执行梯度上升：

$$
\theta
\leftarrow
\theta+
\alpha
\sum_tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

但 PyTorch 优化器默认最小化损失，因此在代码中加入负号：

$$
L_{\text{policy}}
=
-\sum_{t=0}^{T-1}
G_t\log\pi_\theta(A_t\mid S_t)
$$

### 步骤 5：反向传播并更新策略

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如果 $G_t$ 很大，最小化上述损失会提高动作 $A_t$ 的概率；如果 $G_t$ 为负，则会压低该动作概率。

---

## 10. REINFORCE 伪代码

```text
输入：策略网络 πθ(a|s)，学习率 α，折扣因子 γ
初始化策略参数 θ

重复执行每一个 episode：
    初始化轨迹缓存
    获得初始状态 S0

    while 回合未结束：
        从 πθ(·|St) 中采样动作 At
        执行动作 At，获得 Rt+1 和 St+1
        保存 log πθ(At|St) 与 Rt+1
        St ← St+1

    G ← 0
    for t = T-1 到 0：
        G ← Rt+1 + γG
        保存 Gt

    Lpolicy ← -Σt Gt log πθ(At|St)
    对 Lpolicy 反向传播并更新 θ
```

---

## 11. PyTorch 实现框架：离散动作环境

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """用于离散动作空间的策略网络。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)


def choose_action(policy: PolicyNetwork, state):
    """根据策略分布采样动作，并返回该动作的对数概率。"""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy(state_tensor)
    distribution = Categorical(probs)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    return action.item(), log_prob


def compute_returns(rewards, gamma: float) -> torch.Tensor:
    """从后向前计算每一步的折扣回报 G_t。"""
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def reinforce_update(policy, optimizer, log_probs, rewards, gamma: float) -> float:
    returns = compute_returns(rewards, gamma)

    # 可选技巧：对一个 batch 的回报进行标准化，常用于减小数值波动。
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = torch.stack([
        -log_prob * G_t for log_prob, G_t in zip(log_probs, returns)
    ]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

训练循环的核心结构如下：

```python
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(1000):
    state, _ = env.reset()
    rewards = []
    log_probs = []
    done = False

    while not done:
        action, log_prob = choose_action(policy, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    loss = reinforce_update(policy, optimizer, log_probs, rewards, gamma)
    episode_return = sum(rewards)
    print(f"Episode={episode:4d}, Return={episode_return:8.2f}, Loss={loss:8.4f}")
```

代码中最核心的一行是：

```python
-loss_prob * G_t
```

准确的实现变量一般写为：

```python
-log_prob * G_t
```

对应的数学形式是：

$$
L_{\text{policy}}
=
-\sum_tG_t\log\pi_\theta(A_t\mid S_t)
$$

---

## 12. 回报标准化的作用与注意事项

实践中常见写法为：

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

即：

$$
\widehat{G}_t
=
\frac{G_t-\mu_G}{\sigma_G+\varepsilon}
$$

其作用是：

- 减小不同轨迹奖励尺度差异造成的梯度波动；
- 改善数值稳定性；
- 在某些任务中加快优化过程。

但需要注意：若仅对一条很短轨迹进行标准化，可能会扭曲该轨迹中回报的绝对含义。更稳妥的做法通常是收集多个 episode 后，以 batch 方式统一处理回报。

---

## 13. REINFORCE 的主要问题：高方差与信用分配

REINFORCE 使用实际采样回报 $G_t$ 作为动作评价信号。这个估计是无偏的，但可能波动很大。

例如一条轨迹包含：

$$
A_0,A_1,A_2,\ldots,A_9
$$

最终失败可能主要是由于 $A_8$ 的错误，但原始 REINFORCE 会用较差的未来回报去影响前面多个动作的概率更新。这导致：

- 某些本来正确的动作被错误削弱；
- 相同状态动作在不同回合中的更新方向差异很大；
- 训练需要更多轨迹才能收敛。

这个问题称为**信用分配问题**：最终奖励到底应归因于哪些动作。

---

## 14. REINFORCE with Baseline：引入状态基线

为了降低方差，可从回报中减去只依赖状态的基线函数 $b(S_t)$：

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_t
\left(G_t-b(S_t)\right)
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

常用选择为状态价值函数：

$$
b(S_t)=V_\phi(S_t)
$$

于是：

$$
\widehat{A}_t=G_t-V_\phi(S_t)
$$

表示该动作实际回报相比当前状态平均表现的提升量，即优势估计。

策略损失可写为：

$$
L_{\text{actor}}
=
-\sum_t
\left(G_t-V_\phi(S_t)\right)
\log\pi_\theta(A_t\mid S_t)
$$

价值网络以蒙特卡洛回报作为监督目标：

$$
L_{\text{value}}
=
\sum_t
\left(V_\phi(S_t)-G_t\right)^2
$$

### 直观解释

若某状态的平均回报为：

$$
V(S_t)=80
$$

动作执行后实际获得：

$$
G_t=100
$$

则：

$$
G_t-V(S_t)=20>0
$$

说明该动作优于平均策略，应提高概率。

若另一个动作得到：

$$
G_t=60
$$

则：

$$
G_t-V(S_t)=-20<0
$$

即使其绝对回报为正，它也低于该状态的平均水平，因此应降低概率。

---

## 15. REINFORCE with Baseline 与 Actor-Critic 的关系

### REINFORCE with Baseline

通常仍等待整条轨迹结束，用完整蒙特卡洛回报 $G_t$ 来训练基线：

$$
V_\phi(S_t)\approx G_t
$$

因此它仍是蒙特卡洛策略梯度方法。

### Actor-Critic

Actor-Critic 可以使用一步 TD 目标：

$$
R_{t+1}+\gamma V_\phi(S_{t+1})
$$

优势估计可写为：

$$
\widehat{A}_t
=
R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t)
$$

这种更新不必总是等待完整回合结束，一般能降低方差并提高学习效率，但会引入价值估计误差带来的偏差。

---

## 16. 从 REINFORCE 到 PPO 的演进关系

### REINFORCE

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

主要问题：回报方差大，必须等待完整回合。

### REINFORCE with Baseline

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_t\left(G_t-V(S_t)\right)
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

改进：使用状态平均表现作为参照，减少无效波动。

### Actor-Critic

$$
\widehat{\nabla_\theta J(\theta)}
=
\sum_t\widehat{A}_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

改进：Critic 可通过 TD 方式提供更及时的评价。

### PPO

PPO 在 Actor-Critic 基础上限制新旧策略概率比的变化幅度：

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min\left(
 r_t(\theta)\widehat{A}_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}_t
\right)
\right]
$$

其中：

$$
r_t(\theta)
=
\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)}
$$

改进：避免单次更新让策略改变过大，从而提升训练稳定性。

---

## 17. 连续动作空间下的 REINFORCE

离散动作中，策略可用类别分布表示：

$$
A_t\sim\operatorname{Categorical}(p_\theta(S_t))
$$

在机械臂或无人机等连续动作任务中，动作可能满足：

$$
A_t\in\mathbb{R}^n
$$

此时策略网络通常输出高斯分布参数：

$$
\mu_\theta(S_t),\qquad \sigma_\theta(S_t)
$$

然后进行动作采样：

$$
A_t\sim\mathcal{N}\left(\mu_\theta(S_t),\sigma_\theta^2(S_t)\right)
$$

策略损失仍然保持相同结构：

$$
L_{\text{policy}}
=
-\sum_tG_t\log\pi_\theta(A_t\mid S_t)
$$

区别仅在于：离散动作中的 $\log\pi_\theta$ 是类别分布的对数概率，连续动作中的 $\log\pi_\theta$ 是概率密度函数的对数值。

---

## 18. 容易混淆的问题

### 18.1 为什么损失函数前面有负号

理论目标是最大化：

$$
J(\theta)
$$

即使用梯度上升：

$$
\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)
$$

但 PyTorch 的优化器执行梯度下降，因此需要定义：

$$
L=-J
$$

所以代码中写为：

```python
loss = -log_prob * G_t
```

### 18.2 为什么训练阶段不能总选最大概率动作

训练需要探索，因此采用：

$$
A_t\sim\pi_\theta(\cdot\mid S_t)
$$

若训练初期总选概率最大的动作，策略可能过早收敛到次优行为，无法探索更高回报路径。测试阶段则可以按任务需要选择最大概率动作。

### 18.3 为什么原始 REINFORCE 不能每一步立即更新

原始算法需要从当前时刻开始的完整回报：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\cdots
$$

在未来奖励尚未产生时，无法获得真实的蒙特卡洛回报。若要求逐步更新，则通常需要采用 Actor-Critic 或其他 TD 方法。

### 18.4 为什么概率增加对应对数概率变大

由于：

$$
0<\pi_\theta(A_t\mid S_t)\le 1
$$

通常有：

$$
\log\pi_\theta(A_t\mid S_t)\le 0
$$

当高回报动作的概率提高时，其对数概率会向 $0$ 增大。最小化：

$$
-G_t\log\pi_\theta(A_t\mid S_t)
$$

便可在 $G_t>0$ 时促进该动作概率上升。

---

## 19. 优点与缺点总结

| 类别 | 内容 |
|---|---|
| 优点 | 算法结构简单，理论直观，直接学习随机策略，支持连续动作，不需要环境可导 |
| 缺点 | 梯度方差大，样本效率低，需要完整回合，信用分配困难，对超参数敏感 |
| 常见改进 | 回报标准化、加入 baseline、Actor-Critic、GAE、PPO |

---

## 20. 最需要记住的结论

REINFORCE 的训练逻辑可以归纳为五步：

1. 策略网络输出动作概率分布 $\pi_\theta(a\mid s)$；
2. 智能体从分布中采样动作 $A_t$；
3. 回合结束后计算动作之后的回报 $G_t$；
4. 用 $G_t$ 加权动作的对数概率；
5. 提高高回报动作的概率，降低低回报动作的概率。

核心损失函数为：

$$
\boxed{
L_{\text{policy}}
=
-\sum_{t=0}^{T-1}
G_t\log\pi_\theta(A_t\mid S_t)
}
$$

其发展路径可概括为：

$$
\text{REINFORCE}
\rightarrow
\text{REINFORCE with Baseline}
\rightarrow
\text{Actor-Critic}
\rightarrow
\text{PPO}
$$

一句话总结：

> REINFORCE 让策略网络产生动作，让环境回报评价动作，再使用回报修正相应动作的概率。
