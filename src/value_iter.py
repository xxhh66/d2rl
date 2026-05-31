"""
价值迭代核心公式：
1. 贝尔曼最优方程（状态价值函数）：
   V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
2. 贝尔曼最优方程（动作价值函数）：
   Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
3. 价值迭代更新公式：
   V_{k+1}(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
4. 收敛条件：
   ||V_{k+1} - V_k||_∞ < θ
5. 最优策略提取：
   π*(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 1. 价值迭代算法类 ====================

class ValueIteration:
    """
    价值迭代算法 (Value Iteration)
    
    核心思想：直接迭代计算最优价值函数，然后提取最优策略
    与策略迭代的区别：不需要显式的策略评估步骤
    """
    
    def __init__(self, 
                 states,           # 状态列表
                 actions,          # 动作列表
                 transition_prob,  # 转移概率 P(s'|s,a)
                 reward_func,      # 奖励函数 R(s,a,s')
                 gamma=0.9,        # 折扣因子
                 theta=1e-6,       # 收敛阈值
                 max_iter=1000):   # 最大迭代次数
        
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        
        # 构建转移概率矩阵 P[s][a][s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for (s, a, s_next), prob in transition_prob.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            s_next_idx = self.states.index(s_next)
            self.P[s_idx, a_idx, s_next_idx] = prob
        
        # 构建奖励函数 R[s][a][s']
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        for (s, a, s_next), reward in reward_func.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            s_next_idx = self.states.index(s_next)
            self.R[s_idx, a_idx, s_next_idx] = reward
        
        # 存储训练历史
        self.V_history = []
        self.policy_history = []
    
    def value_iteration(self, verbose=True):
        """
        价值迭代主算法
        
        算法步骤：
        1. 初始化 V(s) = 0
        2. 对每个状态，应用贝尔曼最优方程更新
        3. 重复直到收敛
        4. 提取最优策略
        """
        # 初始化价值函数
        V = np.zeros(self.n_states)
        self.V_history = [V.copy()]
        
        if verbose:
            print("\n" + "=" * 70)
            print("价值迭代开始")
            print("=" * 70)
            print(f"状态数: {self.n_states}, 动作数: {self.n_actions}")
            print(f"折扣因子 γ: {self.gamma}, 收敛阈值 θ: {self.theta}")
            print("-" * 70)
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            V_new = np.zeros(self.n_states)
            max_diff = 0
            
            # 对每个状态进行更新
            for s in range(self.n_states):
                # 贝尔曼最优方程：V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
                action_values = []
                
                for a in range(self.n_actions):
                    # 计算动作价值 Q(s,a)
                    q_value = self._compute_q_value(s, a, V)
                    action_values.append(q_value)
                
                # 选择最大值
                V_new[s] = max(action_values)
                
                # 记录最大变化
                max_diff = max(max_diff, abs(V_new[s] - V[s]))
            
            self.V_history.append(V_new.copy())
            
            # 检查收敛
            if max_diff < self.theta:
                if verbose:
                    print(f"\n✓ 收敛于第 {iteration+1} 次迭代")
                    print(f"  最大变化: {max_diff:.2e}")
                    print(f"  运行时间: {time.time() - start_time:.3f} 秒")
                break
            
            V = V_new
            
            # 打印进度
            if verbose and (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration+1:4d}: 最大变化={max_diff:.2e}")
        
        # 提取最优策略
        optimal_policy = self._extract_optimal_policy(V)
        
        # 计算最优价值函数字典
        optimal_values = {self.states[i]: V[i] for i in range(self.n_states)}
        
        return optimal_policy, optimal_values, V
    
    def _compute_q_value(self, s, a, V):
        """
        计算动作价值 Q(s,a)
        
        公式：Q(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V(s')
        """
        # 期望即时奖励
        expected_reward = np.sum(self.P[s, a] * self.R[s, a])
        
        # 期望未来奖励
        expected_future = np.sum(self.P[s, a] * V)
        
        # Q值
        q_value = expected_reward + self.gamma * expected_future
        
        return q_value
    
    def _extract_optimal_policy(self, V):
        """
        从最优价值函数提取最优策略
        
        公式：π*(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
        """
        optimal_policy = {}
        
        for s_idx, state in enumerate(self.states):
            # 计算每个动作的Q值
            action_values = []
            for a_idx, action in enumerate(self.actions):
                q_value = self._compute_q_value(s_idx, a_idx, V)
                action_values.append((q_value, action, a_idx))
            
            # 选择最优动作
            max_q_value = max(action_values, key=lambda x: x[0])
            
            # 确定性策略：最优动作概率为1
            for action in self.actions:
                if action == max_q_value[1]:
                    optimal_policy[(state, action)] = 1.0
                else:
                    optimal_policy[(state, action)] = 0.0
        
        return optimal_policy
    
    def value_iteration_with_early_stop(self, early_stop_threshold=100):
        """
        带早停机制的价值迭代
        """
        V = np.zeros(self.n_states)
        no_improvement_count = 0
        best_policy = None
        
        for iteration in range(self.max_iter):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                action_values = [self._compute_q_value(s, a, V) for a in range(self.n_actions)]
                V_new[s] = max(action_values)
            
            # 检查策略是否变化
            current_policy = self._extract_optimal_policy(V_new)
            if best_policy is not None and current_policy == best_policy:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    print(f"策略稳定，早停于第 {iteration+1} 次迭代")
                    break
            else:
                no_improvement_count = 0
                best_policy = current_policy
            
            if np.max(np.abs(V_new - V)) < self.theta:
                break
            
            V = V_new
        
        return self._extract_optimal_policy(V), {self.states[i]: V[i] for i in range(self.n_states)}, V
    
    def asynchronous_value_iteration(self, update_order='random'):
        """
        异步价值迭代：每次只更新部分状态
        
        Args:
            update_order: 更新顺序 ('random', 'priority', 'sweep')
        """
        V = np.zeros(self.n_states)
        
        print("\n异步价值迭代...")
        
        for iteration in range(self.max_iter):
            V_old = V.copy()
            
            # 确定更新顺序
            if update_order == 'random':
                update_states = np.random.permutation(self.n_states)
            elif update_order == 'priority':
                # 优先更新变化大的状态
                changes = np.abs(V - V_old)
                update_states = np.argsort(changes)[::-1]
            else:  # sweep
                update_states = range(self.n_states)
            
            # 更新选中的状态
            for s in update_states:
                action_values = [self._compute_q_value(s, a, V) for a in range(self.n_actions)]
                V[s] = max(action_values)
            
            # 检查收敛
            if np.max(np.abs(V - V_old)) < self.theta:
                print(f"收敛于第 {iteration+1} 次迭代")
                break
        
        return self._extract_optimal_policy(V), {self.states[i]: V[i] for i in range(self.n_states)}, V
    
    def visualize_value_function(self, V, title="最优价值函数"):
        """可视化价值函数"""
        states = list(V.keys())
        values = list(V.values())
        
        plt.figure(figsize=(12, 5))
        
        # 条形图
        plt.subplot(1, 2, 1)
        bars = plt.bar(states, values, color='lightcoral', edgecolor='darkred', alpha=0.7)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('状态')
        plt.ylabel('最优价值 V*(s)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 热力图（适用于网格世界）
        plt.subplot(1, 2, 2)
        if len(states) <= 16:
            grid_size = int(np.sqrt(len(states)))
            value_grid = np.array(values).reshape(grid_size, grid_size)
            im = plt.imshow(value_grid, cmap='RdYlGn', interpolation='nearest')
            plt.colorbar(im, label='最优价值')
            plt.title('最优价值热力图', fontsize=14, fontweight='bold')
            
            for i in range(grid_size):
                for j in range(grid_size):
                    plt.text(j, i, f'{value_grid[i, j]:.1f}',
                            ha='center', va='center', 
                            color='black' if value_grid[i, j] < 50 else 'white')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_convergence(self):
        """可视化价值迭代收敛过程"""
        V_history = np.array(self.V_history)
        
        plt.figure(figsize=(12, 5))
        
        # 价值收敛曲线
        plt.subplot(1, 2, 1)
        n_plot = min(5, self.n_states)
        for i in range(n_plot):
            plt.plot(V_history[:, i], label=f'状态 {self.states[i]}', linewidth=1.5)
        plt.xlabel('迭代次数')
        plt.ylabel('价值 V(s)')
        plt.title('价值迭代收敛过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 变化量曲线
        plt.subplot(1, 2, 2)
        changes = [np.max(np.abs(V_history[i] - V_history[i-1])) 
                   for i in range(1, len(V_history))]
        plt.semilogy(changes, linewidth=1.5)
        plt.xlabel('迭代次数')
        plt.ylabel('最大变化 (对数尺度)')
        plt.title('收敛速度', fontsize=14, fontweight='bold')
        plt.axhline(y=self.theta, color='r', linestyle='--', label=f'阈值 θ={self.theta}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_optimal_policy(self, policy):
        """可视化最优策略"""
        policy_matrix = np.zeros((self.n_states, self.n_actions))
        for (s, a), prob in policy.items():
            s_idx = self.states.index(s)
            a_idx = self.actions.index(a)
            policy_matrix[s_idx, a_idx] = prob
        
        plt.figure(figsize=(10, 6))
        plt.imshow(policy_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(label='策略概率')
        plt.title('最优策略', fontsize=14, fontweight='bold')
        plt.xlabel('动作')
        plt.ylabel('状态')
        plt.xticks(range(self.n_actions), self.actions)
        plt.yticks(range(self.n_states), self.states)
        
        for i in range(self.n_states):
            for j in range(self.n_actions):
                plt.text(j, i, f'{policy_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color='black' if policy_matrix[i, j] < 0.5 else 'white')
        
        plt.tight_layout()
        plt.show()


# ==================== 2. 示例：网格世界 ====================

def example_grid_world_vi():
    """网格世界示例"""
    print("\n" + "=" * 70)
    print("示例1：4x4网格世界 - 价值迭代")
    print("=" * 70)
    
    # 定义状态
    states = [f'({i},{j})' for i in range(4) for j in range(4)]
    
    # 定义动作
    actions = ['上', '下', '左', '右']
    
    # 构建转移和奖励
    transition_prob = {}
    reward_func = {}
    
    for i in range(4):
        for j in range(4):
            state = f'({i},{j})'
            is_goal = (i == 3 and j == 3)
            is_hole = (i == 1 and j == 1) or (i == 1 and j == 2)  # 陷阱
            
            for action in actions:
                # 计算下一个位置
                ni, nj = i, j
                if action == '上' and i > 0:
                    ni = i - 1
                elif action == '下' and i < 3:
                    ni = i + 1
                elif action == '左' and j > 0:
                    nj = j - 1
                elif action == '右' and j < 3:
                    nj = j + 1
                
                next_state = f'({ni},{nj})'
                transition_prob[(state, action, next_state)] = 1.0
                
                # 设置奖励
                if is_goal:
                    reward = 0
                elif next_state == '(3,3)':
                    reward = 10
                elif is_hole:
                    reward = -5
                else:
                    reward = -0.1
                
                reward_func[(state, action, next_state)] = reward
    
    # 创建价值迭代对象
    vi = ValueIteration(states, actions, transition_prob, reward_func, gamma=0.95, theta=1e-6)
    
    # 执行价值迭代
    policy, values, V = vi.value_iteration(verbose=True)
    
    # 显示结果
    print("\n" + "-" * 70)
    print("最优价值函数:")
    for i in range(4):
        row = []
        for j in range(4):
            state = f'({i},{j})'
            row.append(f"{values[state]:6.2f}")
        print("  " + "  ".join(row))
    
    print("\n最优策略:")
    for i in range(4):
        row = []
        for j in range(4):
            state = f'({i},{j})'
            for action in actions:
                if policy.get((state, action), 0) > 0:
                    row.append(action[0])  # 取动作首字母
                    break
        print("  " + "  ".join(row))
    
    # 可视化
    vi.visualize_value_function(values, "网格世界最优价值函数")
    vi.visualize_convergence()
    vi.visualize_optimal_policy(policy)


# ==================== 3. 示例：简单MDP ====================

def example_simple_mdp_vi():
    """简单MDP示例"""
    print("\n" + "=" * 70)
    print("示例2：简单MDP - 价值迭代详细演示")
    print("=" * 70)
    
    # 定义状态和动作
    states = ['S1', 'S2', 'S3']
    actions = ['a1', 'a2']
    
    # 转移概率
    transition_prob = {
        ('S1', 'a1', 'S1'): 0.5, ('S1', 'a1', 'S2'): 0.5,
        ('S1', 'a2', 'S2'): 0.7, ('S1', 'a2', 'S3'): 0.3,
        ('S2', 'a1', 'S1'): 0.3, ('S2', 'a1', 'S2'): 0.7,
        ('S2', 'a2', 'S3'): 1.0,
        ('S3', 'a1', 'S3'): 1.0,
        ('S3', 'a2', 'S3'): 1.0,
    }
    
    # 奖励函数
    reward_func = {
        ('S1', 'a1', 'S1'): 0, ('S1', 'a1', 'S2'): 1,
        ('S1', 'a2', 'S2'): 1, ('S1', 'a2', 'S3'): 2,
        ('S2', 'a1', 'S1'): 0, ('S2', 'a1', 'S2'): 1,
        ('S2', 'a2', 'S3'): 2,
        ('S3', 'a1', 'S3'): 0,
        ('S3', 'a2', 'S3'): 0,
    }
    
    # 创建价值迭代对象
    vi = ValueIteration(states, actions, transition_prob, reward_func, gamma=0.9, theta=1e-8)
    
    # 执行价值迭代
    policy, values, V = vi.value_iteration(verbose=True)
    
    print("\n" + "-" * 70)
    print("结果:")
    print(f"最优价值函数: {values}")
    print(f"最优策略: {policy}")
    
    # 演示迭代过程
    print("\n" + "-" * 70)
    print("价值迭代过程（前10次迭代）:")
    print("-" * 70)
    
    # 手动演示前几次迭代
    V = np.zeros(len(states))
    for iteration in range(10):
        V_new = np.zeros(len(states))
        for s in range(len(states)):
            q_values = []
            for a in range(len(actions)):
                q = vi._compute_q_value(s, a, V)
                q_values.append(q)
            V_new[s] = max(q_values)
        
        print(f"迭代 {iteration+1:2d}: V = [{V_new[0]:.4f}, {V_new[1]:.4f}, {V_new[2]:.4f}]")
        V = V_new


# ==================== 4. 价值迭代 vs 策略迭代对比 ====================

def compare_value_vs_policy_iteration():
    """对比价值迭代和策略迭代"""
    print("\n" + "=" * 70)
    print("价值迭代 vs 策略迭代 对比")
    print("=" * 70)
    
    # 定义相同的问题
    states = ['A', 'B', 'C']
    actions = ['a1', 'a2']
    
    transition_prob = {
        ('A', 'a1', 'A'): 0.5, ('A', 'a1', 'B'): 0.5,
        ('A', 'a2', 'B'): 0.7, ('A', 'a2', 'C'): 0.3,
        ('B', 'a1', 'A'): 0.3, ('B', 'a1', 'B'): 0.7,
        ('B', 'a2', 'C'): 1.0,
        ('C', 'a1', 'C'): 1.0,
        ('C', 'a2', 'C'): 1.0,
    }
    
    reward_func = {
        ('A', 'a1', 'A'): 0, ('A', 'a1', 'B'): 1,
        ('A', 'a2', 'B'): 1, ('A', 'a2', 'C'): 2,
        ('B', 'a1', 'A'): 0, ('B', 'a1', 'B'): 1,
        ('B', 'a2', 'C'): 2,
        ('C', 'a1', 'C'): 0,
        ('C', 'a2', 'C'): 0,
    }
    
    # 价值迭代
    print("\n" + "-" * 50)
    print("价值迭代:")
    print("-" * 50)
    
    vi = ValueIteration(states, actions, transition_prob, reward_func, gamma=0.9, theta=1e-8)
    
    start_time = time.time()
    policy_vi, values_vi, _ = vi.value_iteration(verbose=False)
    vi_time = time.time() - start_time
    
    print(f"执行时间: {vi_time*1000:.3f} ms")
    print(f"最优价值: {values_vi}")
    
    # 策略迭代（简化版）
    print("\n" + "-" * 50)
    print("策略迭代:")
    print("-" * 50)
    
    from policy_eva import PolicyEvaluator
    
    class SimplePolicyIteration:
        def __init__(self, states, actions, transition_prob, reward_func, gamma=0.9):
            self.states = states
            self.actions = actions
            self.gamma = gamma
            self.evaluator = PolicyEvaluator(states, actions, transition_prob, reward_func, gamma)
        
        def policy_iteration(self):
            # 初始化随机策略
            policy = {}
            for s in self.states:
                for a in self.actions:
                    policy[(s, a)] = 1.0 / len(self.actions)
            
            for iteration in range(100):
                # 策略评估
                V, _ = self.evaluator.policy_evaluation(policy, method='iterative')
                
                # 策略提升
                new_policy = {}
                for s in self.states:
                    best_action = None
                    best_value = -np.inf
                    
                    for a in self.actions:
                        # 计算Q值（简化）
                        q_value = 0
                        for s_next in self.states:
                            prob = transition_prob.get((s, a, s_next), 0)
                            reward = reward_func.get((s, a, s_next), 0)
                            q_value += prob * (reward + self.gamma * V.get(s_next, 0))
                        
                        if q_value > best_value:
                            best_value = q_value
                            best_action = a
                    
                    new_policy[(s, best_action)] = 1.0
                
                if new_policy == policy:
                    break
                policy = new_policy
            
            return policy, V
    
    spi = SimplePolicyIteration(states, actions, transition_prob, reward_func, gamma=0.9)
    
    start_time = time.time()
    policy_pi, values_pi = spi.policy_iteration()
    pi_time = time.time() - start_time
    
    print(f"执行时间: {pi_time*1000:.3f} ms")
    print(f"最优价值: {values_pi}")
    
    print("\n" + "-" * 50)
    print("对比结果:")
    print(f"价值迭代时间: {vi_time*1000:.3f} ms")
    print(f"策略迭代时间: {pi_time*1000:.3f} ms")
    print(f"速度比: {pi_time/vi_time:.2f}x")


# ==================== 5. 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("价值迭代算法 (Value Iteration) 完整实现")
    print("=" * 70)
    
    # 核心公式说明
    print("\n" + "-" * 70)
    print("核心公式:")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  贝尔曼最优方程（价值迭代基础）:                                  │
    │                                                                  │
    │  V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]              │
    │                                                                  │
    │  价值迭代更新公式:                                               │
    │                                                                  │
    │  V_{k+1}(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]        │
    │                                                                  │
    │  收敛条件:                                                       │
    │                                                                  │
    │  ||V_{k+1} - V_k||_∞ < θ                                       │
    │                                                                  │
    │  最优策略提取:                                                   │
    │                                                                  │
    │  π*(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]           │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 运行示例
    example_simple_mdp_vi()
    example_grid_world_vi()
    compare_value_vs_policy_iteration()
    
    print("\n" + "=" * 70)
    print("价值迭代算法总结")
    print("=" * 70)
    print("""
    ┌─────────────────┬────────────────────────────────────────────┐
    │     特点        │                   说明                      │
    ├─────────────────┼────────────────────────────────────────────┤
    │ 收敛速度        │ 线性收敛，通常比策略迭代快                   │
    ├─────────────────┼────────────────────────────────────────────┤
    │ 内存占用        │ O(|S|)，只需要存储价值函数                   │
    ├─────────────────┼────────────────────────────────────────────┤
    │ 适用场景        │ 大规模状态空间，不需要显式策略               │
    ├─────────────────┼────────────────────────────────────────────┤
    │ 优点            │ 实现简单，直接得到最优价值                   │
    ├─────────────────┼────────────────────────────────────────────┤
    │ 缺点            │ 需要知道环境模型（转移概率和奖励）           │
    └─────────────────┴────────────────────────────────────────────┘
    
    与策略迭代的区别：
    - 策略迭代：策略评估 + 策略提升，需要两次迭代
    - 价值迭代：直接更新价值函数，一次迭代完成
    - 价值迭代通常更快收敛到最优策略
    """)