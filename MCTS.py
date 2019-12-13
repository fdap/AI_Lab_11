#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.12.12
#   Intro: 用于黑白棋的MCTS算法
#####################################

# 导入相关模块
import copy
import numpy as np


class MCTS_Node():
    def __init__(self, prob, parent):
        '''
            初始化一个节点
            参数：
                prob: 先验概率
                parent: 父节点
            返回值：
                None
        '''
        self.visit_num = 0
        self.total_value = 0
        self.prob = prob
        self.parent = parent
        self.children = {}


    def get_value(self, c_puct):
        '''
            计算当前节点的分数
            参数：
                c_puct: 控制探索程度
            返回值：
                Q + U: 计算结果
        '''
        if self.visit_num == 0:
            Q = 0
        else:
            Q = self.total_value / self.visit_num
        U = c_puct * self.prob * np.sqrt(self.parent.visit_num) / (1 + self.visit_num)
        
        return Q + U


    def update(self, value):
        '''
            更新访问次数和value
            参数：
                value: 价值
            返回值：
                None
        '''
        self.total_value += value
        self.visit_num += 1


    def select(self, c_puct):
        '''
            根据公式选择子节点
            参数：
                c_puct: 控制探索程度
            返回值：
                next_step, next_node: 落子，相应的子节点
        '''
        next_step, next_node = max(self.children.items(), key=lambda x: x[1].get_value(c_puct))
        return next_step, next_node


    def expand(self, action_prob):
        '''
            expand步骤
            参数：
                action_prob: {action: 概率}
            返回值：
                None
        '''
        if action_prob:
            for action, prob in action_prob.items():
                self.children[action] = MCTS_Node(prob, self)
        else:
            self.children[None] = MCTS_Node(1, self)


class MCTS():
    def __init__(self, sim_num, c_puct, policy_value_fn):
        '''
            初始化函数
            参数：
                sim_num: 执行一次搜索需要模拟的次数
                c_puct: 控制探索程度
                policy_value_fn: 根据一个游戏实例返回策略和价值的函数
            返回值：
                None
        '''
        self.sim_num = sim_num
        self.c_puct = c_puct
        self.policy_value_fn = policy_value_fn
        self.root = MCTS_Node(1.0, None)

    
    def simulation(self, game):
        '''
            一次模拟过程，包含selct, expand, backup三个步骤
            参数：
                game: 游戏实例，注意复制
            返回值：
                None
        '''
        cur_node = self.root
        
        # select
        while cur_node.children:
            next_step, cur_node = cur_node.select(self.c_puct)
            if next_step:
                game.step(next_step[0], next_step[1])

        # expand
        policy, value = self.policy_value_fn(copy.deepcopy(game))
        if game.if_gameover:
            # 游戏结束
            count_1, count_2 = game.get_state()
            if (count_1 > count_2 and game.turn == 1) or \
                (count_1 < count_2 and game.turn == -1):
                value = 1
            else:
                value = -1
        else:
            # 游戏尚未结束
            cur_node.expand(policy)

        # backup
        while cur_node.parent:
            cur_node.update(value)
            value = -value
            cur_node = cur_node.parent

        self.root.update(value)

    
    def play(self):
        '''
            play步骤，决定落子
            参数：
                None
            返回值：
                选择的落子位置，相应的子节点
        '''
        return max(self.root.children.items(), key=lambda x: x[1].visit_num)

    
    def play_with_noise(self):
        '''
            play步骤，决定落子，加入随机过程
            参数：
                None
            返回值：
                policy, next_step, next_node: 概率分布，选择的落子位置，相应子节点
        '''
        policy = {action: node.visit_num / self.root.visit_num \
            for action, node in self.root.children.items()}
        probs = list(policy.values())
        probs[-1] += 1 - sum(probs)
        idx = np.random.choice(len(policy), p=probs)
        next_step = list(policy.keys())[idx]
        next_node = self.root.children[next_step]

        return policy, next_step, next_node

    
    def search(self, game):
        '''
            总搜索流程，实际应用
            参数：
                game: 游戏实例
            返回值：
                row, col: 选择的落子位置
        '''
        # 更新根节点
        if game.turn == 1:
            for step in game.record_2:
                if self.root.children:
                    self.root = self.root.children[step]
                    while None in self.root.children:
                        self.root = self.root.children[None]
                else:
                    self.root = MCTS_Node(1.0, None)
                    break
        else:
            for step in game.record_1:
                if self.root.children:
                    self.root = self.root.children[step]
                    while None in self.root.children:
                        self.root = self.root.children[None]
                else:
                    self.root = MCTS_Node(1.0, None)
                    break
        
        # simulation
        for _ in range(self.sim_num):
            self.simulation(copy.deepcopy(game))

        # play
        next_step, self.root = self.play()
    
        # pass
        while None in self.root.children:
            self.root = self.root.children[None]

        return next_step[0], next_step[1]


    def train_search(self, game):
        '''
            总搜索流程，用于训练
            参数：
                game: 游戏实例
            返回值：
                policy, next_step: 概率分布，选择的落子位置
        '''
        # simulation
        for _ in range(self.sim_num):
            self.simulation(copy.deepcopy(game))

        # play
        policy, next_step, self.root = self.play_with_noise()
    
        # pass
        while None in self.root.children:
            self.root = self.root.children[None]

        return policy, next_step

    
    def self_play(self, game):
        '''
            进行自我对弈
            参数：
                game: 游戏实例
            返回值：
                experience: [(state, policy, turn, value)]
        '''
        state_list = []
        policy_list = []
        turn_list = []

        # 进行对弈
        while game.if_gameover == False:
            # 记录棋局
            state_list.append(game.board.copy())
            # 记录执棋方
            turn_list.append(game.turn)
            # 进行搜索
            policy, next_step = self.train_search(game)
            # 记录策略
            policy_list.append(policy)
            # 落子
            game.step(next_step[0], next_step[1])
            # 自动pass
            while None in self.root.children:
                self.root = self.root.children[None]

        # 胜负判定
        count_1, count_2 = game.get_state()
        value = 1 if count_1 > count_2 else -1

        # 处理成经验
        experience = [(state_list[i], policy_list[i], turn_list[i], value) \
            for i in range(len(turn_list))]

        return experience




def roll_out(game):
    '''
        rollout函数
        参数：
            game: 游戏实例
        返回值：
            policy, value: 策略，价值
    '''
    # 随机策略
    policy = {}
    if game.step_list:
        for step in game.step_list.keys():
            policy[step] = 1 / len(game.step_list)

    # 随机落子到结局
    cur_turn = game.turn
    while game.if_gameover == False:
        next_step = list(game.step_list.keys())[np.random.randint(len(game.step_list))]
        game.step(next_step[0], next_step[1])

    # 判定胜负
    count_1, count_2 = game.get_state()
    if (count_1 > count_2 and cur_turn == 1) or \
        (count_1 < count_2 and cur_turn == -1):
        value = 1
    else:
        value = -1

    return policy, value