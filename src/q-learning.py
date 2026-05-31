"""
算法：
MC->TD->Q-Learning->SARSA->REINFORCE->Actor-Critic->PPO
Q-Learning 算法实现 - FrozenLake 环境
理论基础：表格型强化学习，使用 Q 表存储状态-动作价值
Q-Learning 核心更新公式

理论公式：
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

其中：
- TD目标: TD_target = r + γ * max_a' Q(s', a')
- TD误差: TD_error = TD_target - Q(s, a)
- 更新: Q(s,a) += α * TD_error
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# ==================== 1. Q-Learning Agent ====================
class QLearningAgent:
    """
    Q-Learning 智能体
    使用 Q 表存储每个状态-动作对的价值
    """
    def __init__(self, 
                 state_space_size,    # 状态空间大小
                 action_space_size,   # 动作空间大小
                 learning_rate=0.1,   # 学习率 α
                 gamma=0.99,          # 折扣因子 γ
                 epsilon_start=1.0,   # 初始探索率
                 epsilon_end=0.01,    # 最小探索率
                 epsilon_decay=0.995  # 探索率衰减因子
                 ):
        """
        初始化 Q-Learning Agent
        
        理论公式对应：
        α = learning_rate     - 控制更新步长
        γ = gamma             - 未来奖励的重要性
        ε = epsilon           - 探索-利用平衡
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate      # α: 学习率
        self.gamma = gamma           # γ: 折扣因子
        self.epsilon = epsilon_start # ε: 当前探索率
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 初始化 Q 表：全零矩阵 [状态数 × 动作数]
        # Q(s, a) 表示在状态 s 采取动作 a 的预期回报
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # 记录训练统计
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_history': []
        }
    
    def choose_action(self, state, eval_mode=False):
        """
        ε-贪婪策略选择动作
        
        数学公式：
        π(a|s) = {
            1 - ε + ε/|A|,  if a = argmax Q(s, a)
            ε/|A|,          otherwise
        }
        
        简化实现：
        a = {
            random action,    with probability ε
            argmax Q(s, a),   with probability 1-ε
        }
        
        Args:
            state: 当前状态 (整数索引)
            eval_mode: 是否评估模式 (True: 不探索)
        Returns:
            选择的动作
        """
        if not eval_mode and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.action_space_size)
        else:
            # 利用：选择 Q 值最大的动作
            # 如果有多个最大 Q 值，随机选择一个
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # 找出所有最大 Q 值对应的动作
            actions_with_max_q = np.where(q_values == max_q)[0]
            return np.random.choice(actions_with_max_q)
    
    def update_epsilon(self):
        """
        衰减探索率 ε
        
        公式：ε ← ε × decay_rate
        边界：ε ≥ ε_min
        """
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning 核心更新公式
        
        理论公式：
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        
        其中：
        - TD目标: TD_target = r + γ * max_a' Q(s', a')
        - TD误差: TD_error = TD_target - Q(s, a)
        - 更新: Q(s,a) += α * TD_error
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 当前 Q 值
        current_q = self.q_table[state, action]
        
        # 计算 TD 目标
        if done:
            # 终止状态：没有未来奖励
            # TD_target = r
            td_target = reward
        else:
            # 非终止状态：TD_target = r + γ * max_a' Q(s', a')
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q
        
        # TD 误差
        td_error = td_target - current_q
        
        # Q 值更新：Q(s,a) += α * TD_error
        self.q_table[state, action] += self.lr * td_error
    
    def save_q_table(self, filename='q_table.npy'):
        """保存 Q 表"""
        np.save(filename, self.q_table)
        print(f"Q 表已保存到 {filename}")
    
    def load_q_table(self, filename='q_table.npy'):
        """加载 Q 表"""
        self.q_table = np.load(filename)
        print(f"Q 表已从 {filename} 加载")


# ==================== 2. 训练函数 ====================
def train_q_learning(env_name='FrozenLake-v1', 
                     num_episodes=1000,
                     render=False,
                     **agent_kwargs):
    """
    训练 Q-Learning 智能体
    
    Args:
        env_name: 环境名称
        num_episodes: 训练的 episode 数量
        render: 是否渲染
        **agent_kwargs: Agent 超参数
    """
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode='human', is_slippery=False)
    else:
        env = gym.make(env_name, is_slippery=False)
    
    # 获取状态和动作空间大小
    state_space_size = env.observation_space.n      # FrozenLake: 16
    action_space_size = env.action_space.n          # FrozenLake: 4
    
    # 创建 Agent
    agent = QLearningAgent(state_space_size, action_space_size, **agent_kwargs)
    
    # 训练记录
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n开始训练 Q-Learning on {env_name}")
    print(f"状态空间大小: {state_space_size}, 动作空间大小: {action_space_size}")
    print(f"超参数: α={agent.lr}, γ={agent.gamma}, ε_start={agent.epsilon_start}, ε_end={agent.epsilon_end}")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # 初始状态
        total_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 学习更新 Q 值
            agent.learn(state, action, reward, next_state, done)
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            step_count += 1
        
        # 衰减探索率
        agent.update_epsilon()
        
        # 记录结果
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {step_count}")
    
    env.close()
    return agent, episode_rewards, episode_lengths


# ==================== 3. 测试函数 ====================
def test_q_learning(agent, env_name='FrozenLake-v1', num_episodes=10):
    """
    测试训练好的 Q-Learning 智能体
    """
    env = gym.make(env_name, render_mode='human', is_slippery=False)
    success_count = 0
    
    print(f"\n开始测试 Q-Learning on {env_name}")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 评估模式：不探索
            action = agent.choose_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 可选：添加延迟以便观察
            # env.render()
        
        if total_reward > 0:
            success_count += 1
        
        print(f"Test Episode {episode+1:2d}: Reward = {total_reward:.0f}, Steps = {step_count}")
    
    env.close()
    success_rate = success_count / num_episodes
    print("-" * 70)
    print(f"成功率: {success_rate:.2%} ({success_count}/{num_episodes})")
    
    return success_rate


# ==================== 4. 可视化函数 ====================
def visualize_q_table(agent, env_name='FrozenLake-v1'):
    """
    可视化 Q 表
    """
    # 获取环境信息
    env = gym.make(env_name, is_slippery=False)
    
    # FrozenLake 是 4x4 网格
    grid_size = 4
    q_table = agent.q_table
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 最优策略可视化
    ax1 = axes[0]
    policy_grid = np.zeros((grid_size, grid_size), dtype=str)
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    for state in range(q_table.shape[0]):
        row = state // grid_size
        col = state % grid_size
        best_action = np.argmax(q_table[state])
        policy_grid[row, col] = action_symbols[best_action]
    
    # 绘制策略网格
    ax1.set_title('Optimal Policy (最优策略)', fontsize=14)
    ax1.set_xlim(-0.5, grid_size - 0.5)
    ax1.set_ylim(-0.5, grid_size - 0.5)
    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.grid(True, alpha=0.3)
    
    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j
            action = policy_grid[i, j]
            ax1.text(j, grid_size - 1 - i, action, 
                    ha='center', va='center', fontsize=20)
    
    # 标记起点和终点
    ax1.scatter(0, grid_size-1, c='green', s=200, marker='o', label='Start')
    ax1.scatter(grid_size-1, 0, c='red', s=200, marker='*', label='Goal')
    ax1.legend(loc='upper right')
    
    # 2. Q 值热力图（以某个动作为例）
    ax2 = axes[1]
    action_idx = 2  # 右移动作
    q_values_grid = q_table[:, action_idx].reshape(grid_size, grid_size)
    
    im = ax2.imshow(q_values_grid, cmap='YlOrRd', interpolation='nearest')
    ax2.set_title(f'Q-Values for Action: Right (动作: 右移)', fontsize=14)
    ax2.set_xlabel('Column (列)')
    ax2.set_ylabel('Row (行)')
    
    # 添加数值标注
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax2.text(j, i, f'{q_values_grid[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_training_results(episode_rewards, episode_lengths, window=50):
    """
    绘制训练结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 奖励曲线
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.4, label='Raw Reward', color='blue')
    
    # 滑动平均
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, 
                               np.ones(window)/window, 
                               mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), smoothed, 
                'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Q-Learning Training Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 步数曲线
    ax2 = axes[1]
    ax2.plot(episode_lengths, alpha=0.4, label='Steps per Episode', color='green')
    
    if len(episode_lengths) >= window:
        smoothed_steps = np.convolve(episode_lengths, 
                                     np.ones(window)/window, 
                                     mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), smoothed_steps,
                'orange', linewidth=2, label=f'Moving Average (window={window})')
    
    ax2.set_xlabel('Episode ')
    ax2.set_ylabel('Steps')
    ax2.set_title('Q-Learning Training Steps Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 5. 对比实验：不同超参数的影响 ====================
def hyperparameter_comparison():
    """
    对比不同学习率和折扣因子对训练效果的影响
    """
    env_name = 'FrozenLake-v1'
    num_episodes = 500
    
    # 不同超参数组合
    param_sets = [
        {'learning_rate': 0.1, 'gamma': 0.95, 'name': 'α=0.1, γ=0.95'},
        {'learning_rate': 0.5, 'gamma': 0.95, 'name': 'α=0.5, γ=0.95'},
        {'learning_rate': 0.1, 'gamma': 0.99, 'name': 'α=0.1, γ=0.99'},
        {'learning_rate': 0.5, 'gamma': 0.99, 'name': 'α=0.5, γ=0.99'},
    ]
    
    plt.figure(figsize=(12, 6))
    
    for params in param_sets:
        print(f"\n训练: {params['name']}")
        agent, rewards, _ = train_q_learning(
            env_name=env_name,
            num_episodes=num_episodes,
            render=False,
            learning_rate=params['learning_rate'],
            gamma=params['gamma']
        )
        
        # 绘制滑动平均奖励
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, 
                label=params['name'], linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Q-Learning: Impact of Hyperparameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 6. 详细示例：手动演示 Q 表更新 ====================
def manual_q_learning_demo():
    """
    手动演示 Q-Learning 的更新过程
    """
    print("=" * 70)
    print("Q-Learning 手动演示 - 简单示例")
    print("=" * 70)
    
    # 创建一个简单的 3 状态环境
    # 状态: 0, 1, 2
    # 动作: 0 (左), 1 (右)
    # 奖励: 到达状态 2 获得奖励 1
    
    # 初始化 Q 表
    q_table = np.zeros((3, 2))
    learning_rate = 0.1
    gamma = 0.9
    
    print("\n初始 Q 表:")
    print(q_table)
    print("\n" + "-" * 70)
    
    # 演示一个更新步骤
    state = 0
    action = 1  # 向右移动
    reward = 0
    next_state = 1
    done = False
    
    print(f"场景: 状态={state}, 动作={action}, 奖励={reward}, 下一状态={next_state}, done={done}")
    
    # 计算 TD 目标
    current_q = q_table[state, action]
    max_next_q = np.max(q_table[next_state])
    td_target = reward + gamma * max_next_q
    
    # 计算 TD 误差
    td_error = td_target - current_q
    
    # 更新 Q 值
    new_q = current_q + learning_rate * td_error
    q_table[state, action] = new_q
    
    print(f"\n更新计算:")
    print(f"  Current Q(s,a) = {current_q}")
    print(f"  max Q(s',a') = {max_next_q}")
    print(f"  TD Target = r + γ * max Q(s',a') = {reward} + {gamma} * {max_next_q} = {td_target}")
    print(f"  TD Error = {td_target} - {current_q} = {td_error}")
    print(f"  New Q(s,a) = {current_q} + {learning_rate} * {td_error} = {new_q}")
    
    print("\n更新后的 Q 表:")
    print(q_table)
    print("=" * 70)


# ==================== 7. 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Q-Learning Algorithm Implementation")
    print("理论与代码对应讲解")
    print("=" * 70)
    
    # 演示手动更新过程
    manual_q_learning_demo()
    
    # 训练 Q-Learning 智能体
    print("\n\n开始训练 Q-Learning...")
    agent, rewards, lengths = train_q_learning(
        env_name='FrozenLake-v1',
        num_episodes=800,
        render=False,
        learning_rate=0.1,    # α
        gamma=0.99,           # γ
        epsilon_start=1.0,    # ε_start
        epsilon_end=0.01,     # ε_min
        epsilon_decay=0.995   # ε_decay
    )
    
    # 打印训练统计
    print("\n" + "-" * 70)
    print("Training Statistics:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Last 100 episodes avg reward: {np.mean(rewards[-100:]):.3f}")
    print(f"  Best episode reward: {np.max(rewards):.0f}")
    print(f"  Success rate (last 100): {np.mean(rewards[-100:]):.1%}")
    
    # 绘制训练曲线
    plot_training_results(rewards, lengths, window=50)
    
    # 可视化 Q 表
    # visualize_q_table(agent)
    
    # # 保存 Q 表
    # agent.save_q_table('q_learning_frozenlake.npy')
    
    # # 测试智能体
    # success_rate = test_q_learning(agent, num_episodes=10)
    
    # 可选：运行超参数对比实验
    # hyperparameter_comparison()