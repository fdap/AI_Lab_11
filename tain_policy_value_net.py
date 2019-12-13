#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.12.14
#   Intro: 训练策略-价值网络
#####################################


# 导入相关模块
import numpy as np
import random
import time
import pickle
from Othello import *
from MCTS import *
from policy_value_net import *
from collections import deque


class PipeLine():
    def __init__(self, file_path=None, if_gpu=False):
        '''
            初始化函数
            参数：
                file_path: 模型路径
                if_gpu: 是否使用GPU
            返回值：
                None
        '''
        # 设置参数
        self.lr = 1e-3  # 学习率
        self.c_puct = 5  # 控制探索程度
        self.board_size = 8  # 棋盘大小
        self.net_sim_num = 10  # 结合网络的模拟次数
        self.rollout_sim_num = 10  # 使用rollout的模拟次数
        self.exp_pool = deque(maxlen=10000)  # 经验池，用于训练网络
        self.epoch_num = 5  # 每一次局对弈的训练次数
        self.batch_size = 512  # 批次大小
        self.run_num = 2  # 自我对弈的局数
        self.check_freq = 2  # 模型与MCTS对弈的频率
        self.check_num = 20  # 模型与MCTS对弈的局数
        self.file_path = file_path
        self.if_gpu = if_gpu

        # 网络初始化
        self.net = PolicyValueNetwork(self.board_size, None, if_gpu)


    def state2features(self, state, extended_policy, turn):
        '''
            将一个棋局处理成网络输入特征
            参数：
                state: 棋局
                extended_policy: 扩展策略
                turn: 执棋方
            返回值：
                features: 特征
        '''
        features = np.zeros((4, self.board_size, self.board_size))

        # 黑白棋位置
        features[0, state==1] = 1
        features[1, state==-1] = 1

        # 执棋方判定
        features[2] == (turn+1)//2

        # 有效落子位置
        features[3, extended_policy!=0] = 1

        return features


    def get_experience(self):
        ''' 
            进行一场自我对弈并获取经验
            参数：
                None
            返回值：
                None
        '''
        mcts = MCTS(self.net_sim_num, self.c_puct, self.net.application)
        game = Othello()
        game.initialize_game()
        experience = mcts.self_play(game)

        # 加入经验池
        for state, policy, turn, value in experience:
            # 扩展policy
            extended_policy = np.zeros_like(state)
            # 获取特征
            for pos, prob in policy.items():
                extended_policy[pos[0], pos[1]] = prob

            # 旋转
            for i in range(4):
                new_state = np.rot90(state, i)
                new_extended_policy = np.rot90(extended_policy, i)
                features = self.state2features(new_state, new_extended_policy, turn)
                value = turn * value
                self.exp_pool.append((features, new_extended_policy, value))

                # 左右翻转
                new_state = np.fliplr(new_state)
                new_extended_policy = np.fliplr(new_extended_policy)
                features = self.state2features(new_state, new_extended_policy, turn)
                self.exp_pool.append((features, new_extended_policy, value))


    def get_batch_data(self):
        ''' 
            从经验池中获取一个批次的数据
            参数：
                None
            返回值：
                batch: (batch_size, 4, board_size, board_size)，输出
                target_policy: (batch_size, board_size, board_size)，目标策略
                target_value: (batch_size)，目标价值
        '''
        batch_experience = random.sample(self.exp_pool, self.batch_size)
        batch = np.array([i[0] for i in batch_experience])
        target_policy = np.reshape(np.array([i[1] for i in batch_experience]), (-1, self.board_size**2))
        target_value = np.array([i[2] for i in batch_experience])

        return batch, target_policy, target_value


    def run(self, total_loss_path=None, policy_loss_path=None, value_loss_path=None, winner_record_path=None):
        '''
            进行多次的自我的对弈并训练网络
            参数：
                total_loss_path, policy_loss_path, value_loss_path, winner_record_path: 相应损失记录的保存路径
            返回值：
                total_loss_list, policy_loss_list, value_loss_list, winner_record: 相应的损失记录
        '''
        # 读取损失记录
        try:
            with open(total_loss_path, 'rb') as f:
                total_loss_list = pickle.load(f)
            with open(policy_loss_path, 'rb') as f:
                policy_loss_list = pickle.load(f)
            with open(value_loss_path, 'rb') as f:
                value_loss_list = pickle.load(f)
            with open(winner_record_path, 'rb') as f:
                winner_record = pickle.load(f)
        except:
            total_loss_list = []
            policy_loss_list = []
            value_loss_list = []
            winner_record = []

        print('************************  Train Begin  ********************************')
        for i in range(self.run_num):
            # 记录时间
            time_point = time.time()

            # 更新经验
            self.get_experience()

            # 检查经验池是否够大
            if len(self.exp_pool) < self.batch_size:
                continue

            # 训练网络
            batch, target_policy, target_value = self.get_batch_data()
            total_loss, policy_loss, value_loss = \
                self.net.train_step(batch, target_policy, target_value)
            
            # 更新损失记录
            total_loss_list.append(total_loss)
            policy_loss_list.append(policy_loss)
            value_loss_list.append(value_loss)

            # 日志输出
            print('Episode {}, total loss: {}, time cost: {}'.format(i, \
                total_loss, time.time()-time_point))

            if (i+1) % self.check_freq == 0:
                # 进行对弈评价
                win_count = 0
                for _ in range(self.check_num):
                    ai_1 = MCTS(self.net_sim_num, self.c_puct, self.net.application)
                    ai_2 = MCTS(self.rollout_sim_num, self.c_puct, roll_out)
                    game = Othello()
                    winner = game.ai_vs_ai(ai_1, ai_2)
                    if winner == 1:
                        win_count += 1
                winner_record.append(win_count)
                print('*-----------------------------------------------*')
                print('Play with pure MSTC and win with a ratio: {}!'.format(win_count/self.check_num))
                print('*-----------------------------------------------*')

                # 保存模型
                self.net.save_model(self.file_path)
                # 保存损失记录
                try:
                    with open(total_loss_path, 'wb') as f:
                        pickle.dump(total_loss_list, f)
                    with open(policy_loss_path, 'wb') as f:
                        pickle.dump(policy_loss_list, f)
                    with open(value_loss_path, 'wb') as f:
                        pickle.dump(value_loss_list, f)
                    with open(winner_record, 'wb') as f:
                        pickle.dump(winner_record, f)
                except:
                    pass

        print('*************************  Train End  *********************************')
        return total_loss_list, policy_loss_list, value_loss_list


if __name__ == "__main__":
    pipeline = PipeLine('model.pkl')
    pipeline.run('total_loss', 'policy_loss', 'value_loss', 'winner_record')