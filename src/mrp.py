"""
markov_reward_process.py
马尔科夫奖励过程 (Markov Reward Process)
"""

import numpy as np
import matplotlib.pyplot as plt

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MarkovRewardProcess:
    """马尔科夫奖励过程"""
    
    def __init__(self, states, transition_matrix, rewards, gamma=0.9):
        self.states = states
        self.n_states = len(states)
        self.transition_matrix = np.array(transition_matrix)
        self.gamma = gamma
        
        # 处理奖励
        if isinstance(rewards, (list, np.ndarray)):
            self.rewards = np.array(rewards, dtype=float)
        elif isinstance(rewards, dict):
            self.rewards = np.array([rewards[s] for s in states], dtype=float)
        else:
            raise ValueError("rewards 必须是列表或字典")
    
    def get_reward(self, state):
        """获取状态的即时奖励"""
        state_idx = self.states.index(state)
        return self.rewards[state_idx]
    
    def get_next_state(self, state):
        """采样下一个状态"""
        state_idx = self.states.index(state)
        probabilities = self.transition_matrix[state_idx]
        next_idx = np.random.choice(self.n_states, p=probabilities)
        return self.states[next_idx]
    
    def generate_episode(self, start_state, length=10):
        """生成带奖励的状态序列"""
        episode = []
        state = start_state
        
        for _ in range(length):
            reward = self.get_reward(state)
            episode.append((state, reward))
            state = self.get_next_state(state)
        
        return episode
    
    def value_iteration(self, max_iter=1000, tol=1e-6):
        """
        迭代法计算状态价值函数 V(s)
        
        贝尔曼方程：V(s) = R(s) + γ * Σ P(s'|s) * V(s')
        """
        V = np.zeros(self.n_states)
        
        for i in range(max_iter):
            # 计算新价值
            V_new = self.rewards + self.gamma * (self.transition_matrix @ V)
            
            # 检查收敛
            diff = np.max(np.abs(V_new - V))
            if diff < tol:
                print(f"收敛于第 {i+1} 步，变化量={diff:.2e}")
                break
            
            V = V_new
        
        return {self.states[i]: V[i] for i in range(self.n_states)}
    
    def value_analytic(self):
        """解析法计算：V = (I - γP)^{-1} * R"""
        I = np.eye(self.n_states)
        A = I - self.gamma * self.transition_matrix
        V = np.linalg.inv(A) @ self.rewards
        return {self.states[i]: V[i] for i in range(self.n_states)}
    
    def monte_carlo(self, start_state, num_episodes=1000, episode_length=100):
        """蒙特卡洛方法估计价值"""
        returns = {state: [] for state in self.states}
        
        for _ in range(num_episodes):
            episode = self.generate_episode(start_state, episode_length)
            
            G = 0
            for t in range(len(episode)-1, -1, -1):
                state, reward = episode[t]
                G = reward + self.gamma * G
                returns[state].append(G)
        
        values = {state: np.mean(returns[state]) for state in self.states if returns[state]}
        return values
    
    def visualize_values(self):
        """可视化价值函数"""
        V = self.value_iteration()
        
        states = list(V.keys())
        values = list(V.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(states, values, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('马尔科夫奖励过程 - 状态价值函数', fontsize=14, fontweight='bold')
        plt.xlabel('状态', fontsize=12)
        plt.ylabel('价值 V(s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# ==================== 示例运行 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("马尔科夫奖励过程示例 - 学生学习过程")
    print("=" * 60)
    
    # 定义状态
    states = ['开始', '学习', '休息', '考试', '毕业']
    
    # 转移概率矩阵
    transition = [
        [0.0, 0.8, 0.2, 0.0, 0.0],  # 开始
        [0.0, 0.6, 0.2, 0.2, 0.0],  # 学习
        [0.0, 0.7, 0.0, 0.3, 0.0],  # 休息
        [0.0, 0.0, 0.0, 0.0, 1.0],  # 考试 -> 毕业
        [0.0, 0.0, 0.0, 0.0, 1.0]   # 毕业 -> 毕业
    ]
    
    # 奖励函数
    rewards = [0, 1, -1, 10, 100]
    
    print("\n奖励设置:")
    for i, state in enumerate(states):
        print(f"  {state}: {rewards[i]}")
    
    # 创建MRP
    mrp = MarkovRewardProcess(states, transition, rewards, gamma=0.9)
    
    # 生成轨迹
    print("\n学习轨迹（从开始状态，10步）：")
    episode = mrp.generate_episode('开始', 10)
    for state, reward in episode:
        print(f"  {state}: 奖励={reward}")
    
    # 计算状态价值（迭代法）
    print("\n" + "=" * 60)
    print("状态价值函数（迭代法）：")
    values = mrp.value_iteration()
    for state, value in values.items():
        print(f"  {state}: {value:.3f}")
    
    # 解析法计算（更准确）
    print("\n" + "=" * 60)
    print("状态价值函数（解析法 - 精确解）：")
    values_analytic = mrp.value_analytic()
    for state, value in values_analytic.items():
        print(f"  {state}: {value:.3f}")
    
    # 蒙特卡洛估计
    print("\n" + "=" * 60)
    print("蒙特卡洛估计（10000次采样）：")
    mc_values = mrp.monte_carlo('开始', num_episodes=10000, episode_length=100)
    for state, value in mc_values.items():
        print(f"  {state}: {value:.3f}")
    
    # 可视化
    mrp.visualize_values()