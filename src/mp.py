"""
markov_process.py
马尔科夫过程 (Markov Process / Markov Chain)
定义：<S, P> - 状态集合和转移概率矩阵
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class MarkovProcess:
    """马尔科夫过程：无记忆性的随机过程"""
    
    def __init__(self, states, transition_matrix):
        """
        初始化马尔科夫过程
        
        Args:
            states: 状态列表，如 ['Sunny', 'Cloudy', 'Rainy']
            transition_matrix: 转移概率矩阵 P[i][j] = P(next=j | current=i)
        """
        self.states = states
        self.n_states = len(states)
        # 确保转换为 numpy 数组
        self.transition_matrix = np.array(transition_matrix)
        
        # 验证每行概率和为1
        for i in range(self.n_states):
            prob_sum = np.sum(self.transition_matrix[i])
            if not np.isclose(prob_sum, 1.0):
                print(f"警告：状态 {states[i]} 的转移概率和为 {prob_sum}")
    
    def get_next_state(self, current_state):
        """采样下一个状态 - 这就是转移过程的核心！"""
        # 步骤1: 获取当前状态的索引
        idx = self.states.index(current_state)
        # 步骤2: 从转移矩阵中取出当前状态到所有下一状态的概率分布
        probs = self.transition_matrix[idx]  # 例如：[0.7, 0.2, 0.1]
        # 步骤3: 根据概率分布采样下一个状态（关键！）
        next_idx = np.random.choice(self.n_states, p=probs)
        # 步骤4: 返回下一个状态
        return self.states[next_idx]
    
    def generate_sequence(self, start_state, steps=10):
        """生成状态序列"""
        sequence = [start_state]
        current = start_state
        
        for _ in range(steps - 1):
            current = self.get_next_state(current)
            sequence.append(current)
        
        return sequence
    
    def compute_stationary_distribution(self, max_iter=1000, tol=1e-6):
        """
        计算平稳分布 π = π * P
        
        平稳分布的含义：
        - 马尔可夫链运行无限长时间后，状态的概率分布会趋于稳定
        - 这个稳定的分布就叫做平稳分布
        - 满足方程：π = π × P（π是行向量，P是转移矩阵）
        
        数学原理：
        设 π = [π₁, π₂, ..., πₙ] 是平稳分布
        则对于所有状态 j：πⱼ = Σᵢ πᵢ × Pᵢⱼ
        
        迭代求解方法（幂迭代法）：
        1. 从初始分布 π₀ = [1/n, 1/n, ..., 1/n] 开始
        2. 反复应用转移矩阵：π_{k+1} = π_k × P
        3. 直到收敛（分布不再变化）
        
        Args:
            max_iter: 最大迭代次数，防止无限循环（默认1000）
            tol: 收敛容差，当分布变化小于此值时认为收敛（默认1e-6）
        
        Returns:
            字典：{状态名称: 平稳概率}
        
        示例：
            如果有3个状态['晴天','多云','雨天']
            返回：{'晴天': 0.47, '多云': 0.32, '雨天': 0.21}
        """
        
        # 步骤1: 初始化概率分布为均匀分布
        # np.ones(self.n_states) 创建全1数组，例如 [1, 1, 1]
        # 除以 self.n_states 得到均匀分布，例如 [1/3, 1/3, 1/3]
        pi = np.ones(self.n_states) / self.n_states
        # 初始时，假设所有状态等概率出现
        
        # 步骤2: 迭代更新分布，直到收敛
        for iteration in range(max_iter):
            # 步骤2.1: 应用一次转移矩阵，得到新分布
            # @ 是矩阵乘法运算符
            # new_pi = pi × P
            # 新分布 = 旧分布 × 转移矩阵
            new_pi = pi @ self.transition_matrix
            # 例如：π_new[晴天] = π_old[晴天]×0.7 + π_old[多云]×0.3 + π_old[雨天]×0.2
            
            # 步骤2.2: 检查是否收敛
            # np.linalg.norm() 计算向量的欧几里得距离（向量长度）
            # 如果新旧分布之间的差异很小（小于容差），认为已收敛
            if np.linalg.norm(new_pi - pi) < tol:
                # 可选：打印收敛信息（调试用）
                # print(f"平稳分布在 {iteration+1} 步收敛")
                break
            
            # 步骤2.3: 更新分布，继续迭代
            pi = new_pi
        
        # 步骤3: 将numpy数组结果转换为字典格式
        # 遍历所有状态索引，将状态名和概率配对
        return {self.states[i]: pi[i] for i in range(self.n_states)}


    # 添加别名方法，保持兼容性
    def stationary_distribution(self, max_iter=1000, tol=1e-6):
        """
        计算平稳分布的别名方法
        
        这个函数只是为了提供更短的函数名，方便调用
        内部直接调用 compute_stationary_distribution() 实现相同功能
        
        Args:
            max_iter: 最大迭代次数（同上述）
            tol: 收敛容差（同上述）
        
        Returns:
            字典：{状态名称: 平稳概率}（同上述）
        
        使用示例：
            dist1 = mp.compute_stationary_distribution()  # 完整名称
            dist2 = mp.stationary_distribution()          # 简短别名
            # 两者返回相同结果
        """
        return self.compute_stationary_distribution(max_iter, tol)
    
    def visualize(self):
        """可视化状态转移图"""
        G = nx.DiGraph()
        
        # 添加节点和边
        for i, s in enumerate(self.states):
            G.add_node(s)
            for j, t in enumerate(self.states):
                # 使用正确的索引方式访问 numpy 数组
                prob = self.transition_matrix[i, j]  # 现在 transition_matrix 是 numpy 数组
                if prob > 0:
                    G.add_edge(s, t, weight=f'{prob:.2f}')
        
        # 绘图
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=12, font_weight='bold')
        
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title('MC', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# ==================== 简单示例（不需要 networkx） ====================
def simple_demo():
    """简单演示，不需要 networkx"""
    print("=" * 50)
    print("马尔科夫过程简单示例")
    print("=" * 50)
    
    # 定义状态和转移矩阵
    states = ['晴天', '多云', '雨天']
    transition = [
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ]
    
    # 创建马尔科夫过程
    mp = MarkovProcess(states, transition)
    
    # 生成状态序列
    print("\n天气序列（从晴天开始，20天）：")
    sequence = mp.generate_sequence('晴天', 20)
    print(' -> '.join(sequence))
    
    # 计算平稳分布
    print("\n平稳分布（长期天气概率）：")
    dist = mp.compute_stationary_distribution()
    for state, prob in dist.items():
        print(f"  {state}: {prob:.2%}")
    
    # 统计状态频率
    print("\n状态转移统计：")
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            prob = mp.transition_matrix[i, j]
            if prob > 0:
                print(f"  {s1} -> {s2}: {prob:.0%}")


# ==================== 示例运行 ====================
if __name__ == "__main__":
    # 运行简单示例
    simple_demo()
    
    # 如果有 networkx，可以运行可视化
    try:
        print("\n" + "=" * 50)
        print("可视化（需要 networkx 库）")
        print("=" * 50)
        
        states = ['A', 'B', 'C']
        transition = [
            [0.5, 0.5, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.6, 0.4]
        ]
        
        mp = MarkovProcess(states, transition)
        mp.visualize()
        
    except ImportError:
        print("\n注意：需要安装 networkx 才能运行可视化")
        print("安装命令：pip install networkx")
    except Exception as e:
        print(f"\n可视化时出错：{e}")