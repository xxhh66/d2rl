# Q-learning 算法详解与推导

> 本文从马尔可夫决策过程、价值函数、Bellman 最优方程开始，逐步推导 Q-learning 的更新公式，并解释它为什么是 off-policy、为什么使用最大 Q 值、以及它与 SARSA 和 DQN 的关系。

---

## 1. Q-learning 要解决什么问题

强化学习中，智能体与环境不断交互：

$$
S_t \rightarrow A_t \rightarrow R_{t+1} \rightarrow S_{t+1}
$$

智能体的目标是学习一个策略 $\pi$，使长期累计回报最大。

累计折扣回报定义为：

$$
G_t =R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots
$$

其中：

- $R_{t+1}$ 是执行动作 $A_t$ 后获得的奖励；
- $\gamma\in[0,1]$ 是折扣因子；
- $\gamma$ 越接近 $1$，越重视长期收益；
- $\gamma$ 越接近 $0$，越重视即时奖励。

Q-learning 的目标是直接学习最优动作价值函数：

$$
Q^*(s,a)
$$

它表示：

> 在状态 $s$ 下先执行动作 $a$，之后都按照最优策略行动，能够获得的最大期望累计回报。

一旦学到了 $Q^*(s,a)$，最优策略可以直接由贪心选择得到：

$$
\pi^*(s)=\arg\max_a Q^*(s,a)
$$

因此，Q-learning 的核心思想是：

> 不直接学习策略，而是学习每个状态-动作对有多好，再选择 Q 值最大的动作。

---

## 2. 状态价值函数与动作价值函数

### 2.1 状态价值函数

对于策略 $\pi$，状态价值函数定义为：

$$
V^\pi(s)=\mathbb{E}_\pi\left[G_t \mid S_t=s\right]
$$

它表示：从状态 $s$ 出发，之后按照策略 $\pi$ 行动，期望能获得多少累计回报。

### 2.2 动作价值函数

动作价值函数定义为：

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[G_t \mid S_t=s,A_t=a\right]
$$

它表示：在状态 $s$ 下先执行动作 $a$，之后按照策略 $\pi$ 行动，期望能获得多少累计回报。

Q-learning 学习的是动作价值函数，而不是状态价值函数。原因是：如果只知道 $V(s)$，还不能直接判断当前状态下哪个动作更好；如果知道 $Q(s,a)$，就可以直接比较动作。

---

## 3. 从回报定义推导 Bellman 方程

从累计回报开始：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

将后面的部分提出：

$$
G_t=R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\gamma^2R_{t+4}+\cdots\right)
$$

括号中的部分正是下一时刻开始的累计回报：

$$
G_{t+1}=R_{t+2}+\gamma R_{t+3}+\gamma^2R_{t+4}+\cdots
$$

所以：

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

对状态 $S_t=s$、动作 $A_t=a$ 取期望：

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma G_{t+1}
\mid
S_t=s,A_t=a
\right]
$$

而下一步开始，如果继续按照策略 $\pi$ 行动，则：

$$
\mathbb{E}_\pi
\left[
G_{t+1}
\mid
S_{t+1}=s'
\right]
=
V^\pi(s')
$$

因此：

$$
Q^\pi(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s,A_t=a
\right]
$$

又因为：

$$
V^\pi(s')
=
\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
$$

所以：

$$
Q^\pi(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\sum_{a'}
\pi(a'\mid S_{t+1})Q^\pi(S_{t+1},a')
\mid
S_t=s,A_t=a
\right]
$$

这就是策略 $\pi$ 下的 Bellman 期望方程。

---

## 4. Bellman 最优方程

Q-learning 要学习的是最优动作价值函数：

$$
Q^*(s,a)
=
\max_\pi Q^\pi(s,a)
$$

在下一状态 $s'$，如果之后都采用最优策略，那么应该选择使 Q 值最大的动作：

$$
\max_{a'} Q^*(s',a')
$$

因此，最优动作价值函数满足 Bellman 最优方程：

$$
Q^*(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q^*(S_{t+1},a')
\mid
S_t=s,A_t=a
\right]
$$

如果环境转移概率写成显式形式 $P(s',r\mid s,a)$，则可以写为：

$$
Q^*(s,a)
=
\sum_{s',r}
P(s',r\mid s,a)
\left[
r
+
\gamma
\max_{a'}Q^*(s',a')
\right]
$$

这说明最优 Q 值应当等于：

$$
\text{即时奖励}
+
\gamma
\times
\text{下一状态的最优未来价值}
$$

---

## 5. 从 Bellman 最优方程到 Q-learning 更新

Bellman 最优方程给出的是期望形式：

$$
Q^*(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q^*(S_{t+1},a')
\right]
$$

但实际训练时，我们通常不知道环境转移概率，也不能直接计算期望。

一次交互只能得到一个样本：

$$
(S_t,A_t,R_{t+1},S_{t+1})
$$

于是，用这一次样本构造目标值：

$$
Y_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
$$

这个目标称为 TD 目标。

当前估计为：

$$
Q(S_t,A_t)
$$

二者之差称为 TD 误差：

$$
\delta_t
=
Y_t-Q(S_t,A_t)
$$

即：

$$
\delta_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
-
Q(S_t,A_t)
$$

如果 TD 目标大于当前估计，说明当前 Q 值低估了，应调高；如果 TD 目标小于当前估计，说明当前 Q 值高估了，应调低。

因此，用增量形式更新：

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\delta_t
$$

代入 TD 误差：

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
-
Q(S_t,A_t)
\right]
$$

这就是 Q-learning 的核心更新公式。

等价地，也可写为：

$$
Q(S_t,A_t)
\leftarrow
(1-\alpha)Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
\right]
$$

---

## 6. 为什么 Q-learning 是 off-policy

Q-learning 的行为策略和学习目标策略可以不同。

训练时，为了探索，智能体通常使用 $\epsilon$-greedy 策略选动作：

$$
A_t
=
\begin{cases}
\text{随机动作}, & \text{概率为 }\epsilon \\
\arg\max_a Q(S_t,a), & \text{概率为 }1-\epsilon
\end{cases}
$$

这称为行为策略，即实际用来和环境交互的策略。

但是 Q-learning 更新时使用的是：

$$
\max_{a'}Q(S_{t+1},a')
$$

这表示下一状态按照贪心最优动作进行估计。

因此，它学习的目标策略是：

$$
\pi_{\text{target}}(s)
=
\arg\max_a Q(s,a)
$$

即使当前实际采样动作来自 $\epsilon$-greedy，更新目标仍然假设下一步会采取当前 Q 表中最优的动作。

所以：

$$
\boxed{
\text{Q-learning 用探索策略采样，但学习贪心最优策略，因此是 off-policy。}
}
$$

---

## 7. Q-learning 与 SARSA 的区别

SARSA 的更新目标是：

$$
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
$$

其中 $A_{t+1}$ 是下一状态下实际按照当前行为策略采样得到的动作。

SARSA 更新公式为：

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
-
Q(S_t,A_t)
\right]
$$

Q-learning 的更新目标是：

$$
R_{t+1}
+
\gamma \max_{a'}Q(S_{t+1},a')
$$

二者区别如下：

| 算法       | TD 目标                                | 学习类型   | 含义                                  |
| ---------- | -------------------------------------- | ---------- | ------------------------------------- |
| SARSA      | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$    | on-policy  | 学习实际执行的 $\epsilon$-greedy 策略 |
| Q-learning | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ | off-policy | 学习贪心最优策略                      |

直观理解：

- SARSA 比较“保守”，因为它考虑了探索动作可能带来的风险；
- Q-learning 更“乐观”，因为它总是假设下一步会选择当前估计最好的动作。

---

## 8. Q-learning 算法流程

若状态和动作数量都有限，可以维护一张 Q 表：

$$
Q(s,a)
$$

算法流程如下：

```text
初始化 Q(s,a)，通常全部设为 0

对每个 episode：

    初始化状态 S

    while episode 未结束：

        根据 epsilon-greedy 从 Q(S, ·) 中选择动作 A

        执行动作 A，得到奖励 R 和下一状态 S'

        计算 TD 目标：
            Y = R + gamma * max_a' Q(S', a')

        计算 TD 误差：
            delta = Y - Q(S, A)

        更新：
            Q(S, A) = Q(S, A) + alpha * delta

        S = S'
```

若 $S'$ 是终止状态，则没有未来价值：

$$
Y=R
$$

更新为：

$$
Q(S,A)
\leftarrow
Q(S,A)
+
\alpha
\left[
R-Q(S,A)
\right]
$$

---

## 9. 一个简单数值例子

假设当前状态为 $s$，执行动作为 $a$。

当前 Q 值为：

$$
Q(s,a)=5
$$

执行动作后获得奖励：

$$
r=2
$$

进入下一状态 $s'$，下一状态下各动作 Q 值为：

$$
Q(s',a_1)=6
$$

$$
Q(s',a_2)=10
$$

因此：

$$
\max_{a'}Q(s',a')=10
$$

设折扣因子为：

$$
\gamma=0.9
$$

学习率为：

$$
\alpha=0.1
$$

TD 目标为：

$$
Y
=
2+0.9\times10
=
11
$$

TD 误差为：

$$
\delta
=
11-5
=
6
$$

更新后：

$$
Q_{\text{new}}(s,a)
=
5+0.1\times6
=
5.6
$$

因为 TD 目标高于当前估计，所以 $Q(s,a)$ 被调高。

---

## 10. Q-learning 为什么可以逐步逼近最优 Q 值

Q-learning 可以看作对 Bellman 最优算子的随机近似。

定义 Bellman 最优算子：

$$
(\mathcal{T}^*Q)(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma\max_{a'}Q(S_{t+1},a')
\mid S_t=s,A_t=a
\right]
$$

最优 Q 函数是该算子的固定点：

$$
Q^*
=
\mathcal{T}^*Q^*
$$

Q-learning 的目标就是不断让当前估计 $Q$ 接近：

$$
\mathcal{T}^*Q
$$

单步样本目标：

$$
R_{t+1}
+
\gamma\max_{a'}Q(S_{t+1},a')
$$

是 Bellman 最优目标的一个采样估计。

因此，更新：

$$
Q
\leftarrow
Q
+
\alpha
\left[
\text{采样 Bellman 目标}
-
Q
\right]
$$

就是用随机样本逐步逼近 Bellman 最优固定点。

在表格型、有限 MDP 中，如果满足：

- 每个状态-动作对被充分访问；
- 学习率满足适当衰减条件；
- 折扣因子 $\gamma<1$；

则 Q-learning 可以收敛到 $Q^*$。

---

## 11. Q-learning 的探索策略

如果总是选择：

$$
\arg\max_a Q(s,a)
$$

那么智能体可能过早陷入局部最优，无法发现更好的动作。

因此常使用 $\epsilon$-greedy：

$$
A_t
=
\begin{cases}
\text{随机动作}, & \epsilon \\
\arg\max_aQ(S_t,a), & 1-\epsilon
\end{cases}
$$

训练早期通常设置较大的 $\epsilon$，鼓励探索；训练后期逐渐减小 $\epsilon$，更多利用已学到的 Q 值。

常见衰减形式为：

$$
\epsilon_t
=
\epsilon_{\min}
+
(\epsilon_{\max}-\epsilon_{\min})
\exp
\left(
-\frac{t}{\tau}
\right)
$$

---

## 12. Q-learning 的局限性

### 12.1 表格规模限制

表格型 Q-learning 需要存储：

$$
Q(s,a)
$$

如果状态数量或动作数量很大，Q 表会变得不可行。

### 12.2 连续状态无法直接使用 Q 表

例如 CartPole 的状态包含位置、速度、角度和角速度，是连续变量。严格表格型 Q-learning 无法直接为每个连续状态建立表项。

解决方式包括：

- 状态离散化；
- 线性函数近似；
- 神经网络函数近似，即 DQN。

### 12.3 最大化操作可能导致过估计

Q-learning 使用：

$$
\max_{a'}Q(S_{t+1},a')
$$

如果 Q 估计中存在噪声，最大化操作容易选择被高估的动作，导致整体高估。

Double Q-learning 和 Double DQN 正是为了缓解这一问题。

---

## 13. 从 Q-learning 到 DQN

DQN 用神经网络近似 Q 函数：

$$
Q_\theta(s,a)
\approx
Q^*(s,a)
$$

对于离散动作，网络输入状态 $s$，输出所有动作的 Q 值：

$$
Q_\theta(s,\cdot)
=
[Q_\theta(s,a_1),Q_\theta(s,a_2),\ldots,Q_\theta(s,a_n)]
$$

DQN 的目标为：

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)
\max_{a'}
Q_{\theta^-}(S_{t+1},a')
$$

其中：

- $\theta$ 是在线网络参数；
- $\theta^-$ 是目标网络参数；
- $d_t$ 表示是否终止。

损失函数为：

$$
L(\theta)
=
\left[
Q_\theta(S_t,A_t)-y_t
\right]^2
$$

或使用 Huber Loss：

$$
L(\theta)
=
\operatorname{Huber}
\left(
Q_\theta(S_t,A_t)-y_t
\right)
$$

DQN 相比表格型 Q-learning 增加了两个关键技巧：

1. **经验回放**：打破样本强相关性，提高数据利用率；
2. **目标网络**：降低目标值随在线网络快速变化带来的不稳定。

---

## 14. Q-learning、DQN、SARSA 的关系

| 方法       | 价值表示            | TD 目标                                | 策略类型   |
| ---------- | ------------------- | -------------------------------------- | ---------- |
| Q-learning | Q 表                | $r+\gamma\max_{a'}Q(s',a')$            | off-policy |
| SARSA      | Q 表                | $r+\gamma Q(s',a')$                    | on-policy  |
| DQN        | 神经网络 $Q_\theta$ | $r+\gamma\max_{a'}Q_{\theta^-}(s',a')$ | off-policy |

DQN 可以理解为：

$$
\boxed{
\text{用神经网络近似 Q 表的 Q-learning}
}
$$

---

## 15. 必须掌握的核心公式

### 累计折扣回报

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+
\cdots
$$

### 动作价值函数

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t\mid S_t=s,A_t=a
\right]
$$

### Bellman 最优方程

$$
Q^*(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q^*(S_{t+1},a')
\mid
S_t=s,A_t=a
\right]
$$

### TD 目标

$$
Y_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
$$

### TD 误差

$$
\delta_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
-
Q(S_t,A_t)
$$

### Q-learning 更新

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
-
Q(S_t,A_t)
\right]
$$

---

## 16. 总结

Q-learning 是一种基于动作价值函数的经典 off-policy 强化学习算法。它不直接学习策略，而是学习最优动作价值函数 $Q^*(s,a)$。

其推导路径是：

$$
G_t
=
R_{t+1}
+
\gamma G_{t+1}
$$

推出 Bellman 最优方程：

$$
Q^*(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q^*(S_{t+1},a')
\right]
$$

再用一次采样得到 TD 目标：

$$
Y_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
$$

最终得到增量更新：

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
Y_t-Q(S_t,A_t)
\right]
$$

也就是：

$$
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
-
Q(S_t,A_t)
\right]
}
$$

一句话理解：

> **Q-learning 用“即时奖励 + 下一状态最优 Q 值”作为目标，不断修正当前动作的 Q 值，最终逼近最优动作价值函数。**
