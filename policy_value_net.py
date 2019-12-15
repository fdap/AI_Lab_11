#####################################
#   Author: 郭俊楠
#   NetID: 17341045
#   Mail: 529931457@qq.com
#   Date: 2019.11.29
#   Intro: 策略-价值网络
#####################################


# 导入相关模块
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np


class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size=8, file_path=None, if_gpu=False):
        '''
            初始化一个策略-价值网络
            参数：
                board_size: 棋盘大小
                file_path: 模型路径
                if_gpu: 是否使用GPU
            返回值：
                None
        '''
        super().__init__()

        # 参数设置
        self.if_gpu = if_gpu
        self.board_size = board_size
        self.l2 = 1e-4
        self.lr = 1e-3

        # 构建网络
        self._build_network()
        self.double()
        if if_gpu:
            self.cuda()
        if file_path:
            self.load_state_dict(torch.load(file_path))

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)


    def _build_network(self):
        '''
            构建网络
            参数：
                None
            返回值：
                None
        '''
        # 特征提取层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 策略头
        self.policy_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4*self.board_size**2, self.board_size**2)

        # 价值头
        self.value_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2*self.board_size**2, 64)
        self.value_fc2 = nn.Linear(64, 1)

        
    def forward(self, data):
        '''
            网络输出
            参数：
                data: 输入
            返回值：
                policy, value: 各个位置的落子概率以及当前局面价值
        '''
        data = data.double()
        if self.if_gpu:
            data = data.cuda()
        

        # 特征提取层
        common = F.relu(self.conv1(data))
        common = F.relu(self.conv2(common))
        common = F.relu(self.conv3(common))

        # 策略头
        policy = F.relu(self.policy_conv1(common))
        policy = policy.view(-1, 4*self.board_size**2)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # 价值头
        value = F.relu(self.value_conv1(common))
        value = value.view(-1, 2*self.board_size**2)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


    def application(self, game):
        '''
            实际应用
            参数：
                game: 游戏实例
            返回值：
                legal_policy, value: 处理过的网络输出
        '''
        # 从当前棋局中获取输入
        data = torch.zeros((1, 4, self.board_size, self.board_size))
        # 黑白棋位置
        data[0, 0, game.board==1] = 1
        data[0, 1, game.board==-1] = -1
        # 执棋方判定
        data[0, 2] = (game.turn+1)//2
        # 有效落子位置
        for row, col in game.step_list.keys():
            data[0, 3, row, col] = 1

        # 前向传播
        policy, value = self(data)

        # 输出处理
        policy = np.exp(policy.data.cpu().numpy())
        legal_policy = {}
        for row, col in game.step_list.keys():
            legal_policy[(row, col)] = policy[0, row*self.board_size+col]
        value = value.data.cpu().numpy()[0][0]

        return legal_policy, value


    def train_step(self, batch, target_policy, target_value):
        '''
            训练一次
            参数：
                batch: (batch_size, 4, board_size, board_size)，输出
                target_policy: (batch_size, board_size, board_size)，目标策略
                target_value: (batch_size)，目标价值
            返回值：
                总损失值，策略损失值，价值损失值
        '''
        if self.if_gpu:
            batch = torch.from_numpy(batch).double().cuda()
            target_policy = torch.from_numpy(target_policy).double().cuda()
            target_value = torch.from_numpy(target_value).double().cuda()
        else:
            batch = torch.from_numpy(batch).double()
            target_policy = torch.from_numpy(target_policy).double()
            target_value = torch.from_numpy(target_value).double()

        self.optimizer.zero_grad()

        # 获取网络输出
        policy, value = self(batch)

        # 定义损失
        policy_loss = F.mse_loss(value.view(-1), target_value)
        value_loss = -torch.mean(torch.sum(policy * target_policy, 1))
        loss = policy_loss + value_loss

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item(),  policy_loss.item(), value_loss.item()


    def save_model(self, file_path):
        '''
            保存模型
            参数：
                file_path: 保存路径
            返回值：
                None
        '''
        torch.save(self.state_dict(), file_path)