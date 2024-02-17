# MIT License

# Copyright (c) 2023 Qiyun Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os
# 忽略特定类型的警告
import warnings
warnings.filterwarnings("ignore")


# 获取路径下的文件函数
def get_file_list(file_path):
    '''
    file_path:str        # 文件夹路径
    return:list          # 返回对应文件夹下所有对应格式组成的路径列表
    '''
    file_list = glob.glob(os.path.join(file_path,'*csv'))
    
    return file_list


def Set_reward_mapping(df,stage_index):
    '''
    分析不同阶段小鼠在go信号下对小鼠的动作进行奖励编码,
    并将小鼠的行为lick,nolick进行分类标志
    '''
    # R = 0  
    if df.loc[stage_index,'outcome'] ==1:      # Action1
            R = 1.0
            lickSig = 1                        # status flag
    elif df.loc[stage_index,'outcome'] == 2:   # Action2
            R = 0.1
            lickSig = 0
    elif df.loc[stage_index,'outcome'] == 3:   # Action3
            R = 0.1
            lickSig = 0
    elif df.loc[stage_index,'outcome'] == 4:   # Action4
            R = -2.0
            lickSig = 1
    
    return R,lickSig

'''
    df: dataframe 传入数据行为每一次试验,column0:每次试验所给的信号,column1:为每次输出的结果或者动作.
    N_STATES: 所切换的状态的总数,如试验的总次数.
    ALPHA: float 取值为0~1,学习率
    GAMMA: float 取值为0~1,折扣率
    stage_name: 获取Q值的阶段,如stable,uncertain,reverse
    return Q_go: 返回单步更新的 Q_lick,Q_nolick的值
    # 算法原理（贝尔曼方程）
    # Q(s, a) = Q(s, a) + α(r + γ * Q(s', a') - Q(s, a))
'''
def get_Q_list_SARSA(df_go,stagename):

    Q_go = []                         # 创建存储迭代过程的go_Q值空列表

    if stagename == 'stage1':
        # 初始化迭代初值
        Qs_go_lick = 0
        Qs_go_nolick = 0

        ALPHA = 0.4
        GAMMA = 0

    elif stagename == 'stage2':
        Qs_go_lick = 0.98
        Qs_go_nolick = 0.028

        ALPHA = 0.1
        GAMMA = 0

    elif stagename == 'stage3':
        Qs_go_lick = 0.98
        Qs_go_nolick = 0.028
        ALPHA = 0.8
        GAMMA = 0

    Q_go.append([Qs_go_lick,Qs_go_nolick])
    go_N_STATES = np.array(df_go).shape[0]

    for i in range(go_N_STATES-1):
        
        R,lickSig = Set_reward_mapping(df_go,i)
        Qs_go_next_R,next_lickSig = Set_reward_mapping(df_go,i+1)
        
        if lickSig ==1:
            Qs_go_lick = Qs_go_lick + ALPHA*(R + GAMMA * Qs_go_next_R - Qs_go_lick)
        else:
            Qs_go_nolick = Qs_go_nolick + ALPHA*(R + GAMMA * Qs_go_next_R - Qs_go_nolick)

        Q_go.append([Qs_go_lick,Qs_go_nolick])

    return Q_go 


'''
    df: dataframe 传入数据行为每一次试验,column0:每次试验所给的信号,column1:为每次输出的结果或者动作.
    N_STATES: 所切换的状态的总数,如试验的总次数.
    ALPHA: float 取值为0~1,学习率
    GAMMA: float 取值为0~1,折扣率
    stage_name: 获取Q值的阶段,如stable,uncertain,reverse
    return Q_go,Q_nogo: 返回单步更新的 Q_go,Q_nogo 值
    
'''
def get_Q_list_TD0(df_state,alpha_init,alpha_discount,pe_discount):

    # 迭代超参数设置
    THETA = pe_discount
    THETA_p = alpha_discount
    PE = 0

    Q_go = []                         # 创建存储迭代过程的go_Q值空列表
    alpha_list = []                   # 创建存储迭代过程的alpha值空列表
    pre_err = []                      # 创建存储迭代过程的PE值空列表

    Qs_go_lick = 0.98
    Qs_go_nolick = 0.028
    ALPHA = alpha_init
    GAMMA = 0

    Q_go.append([Qs_go_lick,Qs_go_nolick])
    alpha_list.append(ALPHA)
    pre_err.append(PE)

    go_N_STATES = np.array(df_state).shape[0]

    for i in range(go_N_STATES-1):
    # for i in range(go_N_STATES):
        
        R,lickSig = Set_reward_mapping(df_state,i)

        if lickSig ==1:
            PE= R - Qs_go_lick
            ALPHA = THETA_p*ALPHA+THETA*abs(PE)
            Qs_go_lick = Qs_go_lick + ALPHA*(R - Qs_go_lick)
            
        else:
            PE = R - Qs_go_nolick
            ALPHA = THETA_p*ALPHA+THETA*abs(PE)
            Qs_go_nolick = Qs_go_nolick + ALPHA*(R - Qs_go_nolick)

        alpha_list.append(ALPHA)
        pre_err.append(PE)
        Q_go.append([Qs_go_lick,Qs_go_nolick])

    return Q_go,alpha_list,pre_err


