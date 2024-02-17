import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets,QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from Ui_PyTDL import Ui_MainWindow  
import time
import numpy as np
from PyQt5.QtGui import QPixmap

def softmax_update(Q, scale_factor):
    e_Q = np.exp(scale_factor * (Q - np.max(Q)))
    P = e_Q / e_Q.sum(axis=1, keepdims=True)
    return P


def sigmoid_update(Q, scale_factor):
    """Sigmoid function with linear scaling factor"""
    scaled_x = scale_factor * (Q - np.mean(Q))  # 缩放Q值，可根据需求调整
    probabilities = 1 / (1 + np.exp(-scaled_x))
    P = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return P


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self._data.columns[section])
        

class DataVisualizationApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 主菜单按钮接口函数
        self.pushButton_2.clicked.connect(self.set_page_1)
        self.pushButton_3.clicked.connect(self.set_page_2)
        self.pushButton_4.clicked.connect(self.set_page_3)
        self.pushButton_5.clicked.connect(self.set_page_4)
        self.pushButton_6.clicked.connect(self.set_page_5)

        # 表征界面====================================================
        self.loadButton.clicked.connect(self.load_data)
        self.setButton.clicked.connect(self.display_mapping)
        self.saveButton.clicked.connect(self.save_mapping)
        self.get_QButton.clicked.connect(self.get_Q_iterm_value) 
        self.pushButton_11.clicked.connect(self.plot_data)
        self.save_data_Button.clicked.connect(self.save_data)
        self.get_PButton.clicked.connect(self.get_P_iterm_value)
        self.pushButton_15.clicked.connect(self.plot_data_P)
        self.pushButton_8.clicked.connect(self.save_figure_Q)
        self.pushButton_9.clicked.connect(self.save_figure_P)
        # Q 值绘图
        self.figure1 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.verticalLayout_4.addWidget(self.canvas1)
        # P 值绘图
        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.verticalLayout_6.addWidget(self.canvas2)

        self.mapping = {}  # 用于存储奖励值编码映射关系
        self.action_mapping = {}  # 用于存储决策动作分类映射关系
        

        # 预测界面=======================================================
        self.loadButton_2.clicked.connect(self.load_data_1)
        self.setButton_2.clicked.connect(self.display_mapping_1)
        self.saveButton_2.clicked.connect(self.save_mapping_1)
        self.get_QButton_2.clicked.connect(self.get_Q_iterm_value_1)
        self.predictionButton.clicked.connect(self.predict_function)
        self.pushButton_12.clicked.connect(self.save_QP_fig)

        # Q 值绘图
        self.figure3 = plt.figure()
        self.canvas3 = FigureCanvas(self.figure3)
        self.verticalLayout_9.addWidget(self.canvas3)
        # P 值绘图
        self.figure4 = plt.figure()
        self.canvas4 = FigureCanvas(self.figure4)
        self.verticalLayout_10.addWidget(self.canvas4)

        self.pushButton_13.clicked.connect(self.save_data_1)

        # 仿真界面===================================================
        self.loadButton_3.clicked.connect(self.load_data_2)
        self.setButton_3.clicked.connect(self.display_mapping_2)
        self.saveButton_3.clicked.connect(self.save_mapping_2)
        self.generate_Button.clicked.connect(self.generate_function)
        self.preview_Button.clicked.connect(self.display_generate_df)
        self.savegendata_Button.clicked.connect(self.save_data_2)

        # Q 值绘图
        self.figure5 = plt.figure()
        self.canvas5 = FigureCanvas(self.figure5)
        self.verticalLayout_14.addWidget(self.canvas5)
        
        # P 值绘图
        self.figure6 = plt.figure()
        self.canvas6 = FigureCanvas(self.figure6)
        self.verticalLayout_13.addWidget(self.canvas6)

        
# ========================================相关函数定义=================================================
# ======================================定义界面一的函数================================================
    # 定义读取数据
    def load_data(self):
        file_path,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'load File', '','Files (*.csv)') 

        if file_path:
            self.dataFrame = pd.read_csv(file_path)
            # print(self.dataFrame)
            QtWidgets.QMessageBox.information(self, 'Read Successful', 'File read successfully.')

    # 定义奖励以及决策分类映射函数
    def display_dataframe(self, df):
        model = PandasModel(df)
        self.tableView.setModel(model)

    def display_mapping(self):
        self.mappingTable.setRowCount(len(self.dataFrame['outcome'].unique()))
        self.mappingTable.setColumnCount(3)
        self.mappingTable.setHorizontalHeaderLabels(['outcome', 'reward','action_type'])

        for idx, value in enumerate(self.dataFrame['outcome'].unique()):
            value_item = QtWidgets.QTableWidgetItem(str(value))

            self.mappingTable.setItem(idx, 0, value_item)

            if value in self.mapping:
                mapping_value = self.mapping[value]
                mapping_item = QtWidgets.QTableWidgetItem(str(mapping_value))
                self.mappingTable.setItem(idx, 1, mapping_item)

    def save_mapping(self):
        for row in range(self.mappingTable.rowCount()):
            value_item = self.mappingTable.item(row, 0)
            mapping_item = self.mappingTable.item(row, 1)
            mapping_item2 = self.mappingTable.item(row, 2)

            if value_item and mapping_item:
                value = value_item.text()
                mapping_value = float(mapping_item.text())
                mapping_value_2 = int(mapping_item2.text())  # 获取动作类别标签

                self.mapping[int(value)] = mapping_value
                self.action_mapping[int(value)] = mapping_value_2

        if self.dataFrame is not None:
                self.dataFrame['reward'] = self.dataFrame['outcome'].map(self.mapping)
                self.dataFrame['action_type'] = self.dataFrame['outcome'].map(self.action_mapping)
                # 显示带有 reward 列的 DataFrame
                self.display_dataframe(self.dataFrame)
        return self.dataFrame
    

    # 获取用户设置的不同动作的迭代初值以及模型超参数
    def Q_init_input(self):
        input_text = self.Qinit_LineEdit.text()  # 获取文本框中的输入内容
        self.input_Q_init = [float(x) for x in input_text.split(',')]  # 将输入内容解析为数组

        # print("User input array:", input_Q_init)

        # 获取模型的超参数设置
        self.alpha = self.alpha_doubleSpinBox.value()  # 获取用户设置的学习率
        self.alpha_init = self.alpha_init_Edit.text()  
        self.gramma = self.grmma_doubleSpinBox.value()  
        self.theta = self.theta_doubleSpinBox.value() 
        self.theta_p = self.theta_p_doubleSpinBox.value() 
        self.pe_init = self.PElineEdit.text()
    

    # 定义获取模型选择函数
    def get_Q_model_type_1(self):
        return self.comboBox.currentText()
    
    def get_P_model_type_1(self):
        return self.comboBox_2.currentText()

    def get_Q_model_type_2(self):
        return self.comboBox_3.currentText()
    
    def get_P_model_type_2(self):
        return self.comboBox_4.currentText()
    
    def get_Q_model_type_3(self):
        return self.comboBox_6.currentText()
    
    def get_P_model_type_3(self):
        return self.comboBox_5.currentText()
    

    # 定义 Q 值迭代过程的函数
    def get_Q_iterm_value(self):
        '''
        df: dataframe 传入数据行为每一次试验,column0:每次试验所给的信号,column1:为每次输出的结果或者动作.
        N_STATES: 所切换的状态的总数,如试验的总次数.
        ALPHA: float 取值为0~1,学习率
        GAMMA: float 取值为0~1,折扣率
        算法原理（贝尔曼方程）
        Q(s, a) = Q(s, a) + α(r + γ * Q(s', a') - Q(s, a))
        '''
        self.Q_init_input()
        # 迭代超参数设置
        THETA = self.theta
        THETA_p = self.theta_p
        PE = self.pe_init
        ALPHA = self.alpha
        GAMMA = self.gramma
        ALPHA_init = self.alpha_init

        Q_go = []                         # 创建存储迭代过程的go_Q值空列表
        alpha_list = []                   # 创建存储迭代过程的alpha值空列表
        pre_err = []                      # 创建存储迭代过程的PE值空列表

        action_Q_list = self.input_Q_init

        Q_go.append(list(action_Q_list))
        alpha_list.append(ALPHA)
        pre_err.append(PE)
        
        go_N_STATES = np.array(self.dataFrame).shape[0]
        TD_type = self.get_Q_model_type_1()
        print(TD_type)
        for i in range(go_N_STATES-1):   
            
            R,ActionSig = self.dataFrame.loc[i,'reward'],self.dataFrame.loc[i,'action_type']
            next_R,next_ActionSig = self.dataFrame.loc[i+1,'reward'],self.dataFrame.loc[i+1,'action_type']
            if TD_type == 'Auto-alpha-TD(0)':
                # Auto_alpha_TD（0）算法更新方式
                PE= R - action_Q_list[ActionSig-1]
                ALPHA = THETA_p*ALPHA+THETA*abs(PE)
                action_Q_list[ActionSig-1] = action_Q_list[ActionSig-1] + ALPHA*(R - action_Q_list[ActionSig-1])
                Q_go.append(list(action_Q_list))
                alpha_list.append(ALPHA)
                pre_err.append(PE)
            elif TD_type == 'SARSA':
                # SARSA 算法更新方式
                action_Q_list[ActionSig-1] = action_Q_list[ActionSig-1] + ALPHA*(R + GAMMA*next_R - action_Q_list[ActionSig-1])
                Q_go.append(list(action_Q_list))

            else:
                QtWidgets.QMessageBox.information(self, '用户自定义', '请根据需要对底层函数代码进行灵活修改.')
                break


        # print(Q_go)
        # self.Q_go_array = np.array(Q_go)
        self.Q_go_array = Q_go

        Q_action_df = pd.DataFrame(self.Q_go_array, columns=[f'Qaciton{m}' for m in range(len(action_Q_list))])  # Convert to DataFrame

        dff = pd.concat([self.dataFrame,Q_action_df],axis=1)
        # dff.drop(columns='index',axis=1,inplace=True)
        # 增加动态的学习率和PE进去
        if TD_type == 'Auto-alpha-TD(0)':
            dff['Alpha'] = alpha_list
            dff['PE'] = pre_err
            self.Q_dff = dff
        else:
            self.Q_dff = dff
        print(self.Q_dff)


    # 定义 Q 值迭代过程的绘图函数
    def plot_data(self):
        df = self.Q_dff
        self.figure1.clear()
        ax = self.figure1.add_subplot(111)
        ax.plot(df.index, df['Qaciton0'], marker='o',c='#C17F6F')
        ax.set_xlabel('trail by trail')
        ax.set_ylabel('Q_value')
        ax.set_title('Q—vari_trail by trail')
        self.canvas1.draw()


    def save_figure_Q(self):
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save fig File', '', 'jpg Files (*.jpg)')
        if save_path:
            self.figure.savefig1(save_path)
            QtWidgets.QMessageBox.information(self, 'Save Successful', 'figure saved successfully.')


    # 定义P值计算函数
    def get_P_iterm_value(self):
        if self.get_P_model_type_1() == 'Softmax':
            P = softmax_update(self.Q_go_array,self.doubleSpinBox.value())
        elif self.get_P_model_type_1() == 'Sigmoid':
            P = sigmoid_update(self.Q_go_array,self.doubleSpinBox.value())
        else:
            QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码')
            pass 
        prob = pd.DataFrame(P, columns=[f'Paciton{n}' for n in range(len(self.input_Q_init))])
        self.QP_df = pd.concat([self.Q_dff,prob],axis=1)

        print(self.QP_df)

    
    # 定义P值绘图函数
    def plot_data_P(self):
        df = self.QP_df
        self.figure2.clear()
        ax = self.figure2.add_subplot(111)
        ax.plot(df.index, df['Paciton0'], marker='o',c='#C17F6F')
        ax.set_xlabel('trail by trail')
        ax.set_ylabel('P_value')
        ax.set_title('P—vari_trail by trail')
        self.canvas2.draw()


    # 定义P值图保存到本地的函数
    def save_figure_P(self):
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save fig File', '', 'jpg Files (*.jpg)')
        if save_path:
            self.figure2.savefig(save_path)
            QtWidgets.QMessageBox.information(self, 'Save Successful', 'figure saved successfully.')


    # 定义数据保存到本地的功能函数
    def saveScreenshot(self):
        # 捕获当前窗口的内容
        screenshot = QPixmap(self.size())
        self.render(screenshot)

        # 将内容保存到文件
        screenshot.save('screenshot.tiff', 'TIFF')

    def save_data(self):
        self.saveScreenshot()       # 保存运行程序界面的截图
        if self.QP_df is not None:  # 替换需要最终要保存的dataframe
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save CSV File', '', 'CSV Files (*.csv)')
            if save_path:
                self.QP_df.to_csv(save_path, index=False)
                QtWidgets.QMessageBox.information(self, 'Save Successful', 'DataFrame saved successfully.')

    # ======================================定义界面二的函数================================================
    # 定义读取数据
    def load_data_1(self):
        file_path,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'load File', '','Files (*.csv)') 

        if file_path:
            self.dataFrame_1 = pd.read_csv(file_path)
            # print(self.dataFrame)
            QtWidgets.QMessageBox.information(self, 'Read Successful', 'File read successfully.')

    # 定义奖励以及决策分类映射函数
    def display_dataframe_1(self, df):
        model = PandasModel(df)
        self.tableView_2.setModel(model)

    def display_mapping_1(self):
        self.mappingTable_2.setRowCount(len(self.dataFrame_1['outcome'].unique()))
        self.mappingTable_2.setColumnCount(3)
        self.mappingTable_2.setHorizontalHeaderLabels(['outcome', 'reward','action_type'])

        for idx, value in enumerate(self.dataFrame_1['outcome'].unique()):
            value_item = QtWidgets.QTableWidgetItem(str(value))

            self.mappingTable_2.setItem(idx, 0, value_item)

            if value in self.mapping:
                mapping_value = self.mapping[value]
                mapping_item = QtWidgets.QTableWidgetItem(str(mapping_value))
                self.mappingTable_2.setItem(idx, 1, mapping_item)

    def save_mapping_1(self):
        for row in range(self.mappingTable_2.rowCount()):
            value_item = self.mappingTable_2.item(row, 0)
            mapping_item = self.mappingTable_2.item(row, 1)
            mapping_item2 = self.mappingTable_2.item(row, 2)

            if value_item and mapping_item:
                value = value_item.text()
                mapping_value = float(mapping_item.text())
                mapping_value_2 = int(mapping_item2.text())  # 获取动作类别标签

                self.mapping[int(value)] = mapping_value
                self.action_mapping[int(value)] = mapping_value_2

        if self.dataFrame_1 is not None:
                self.dataFrame_1['reward'] = self.dataFrame_1['outcome'].map(self.mapping)
                self.dataFrame_1['action_type'] = self.dataFrame_1['outcome'].map(self.action_mapping)
                # 显示带有 reward 列的 DataFrame
                self.display_dataframe_1(self.dataFrame_1)
        return self.dataFrame_1
    

    # 获取用户设置的不同动作的迭代初值以及模型超参数
    def Q_init_input_1(self):
        input_text = self.Qinit_LineEdit_2.text()  # 获取文本框中的输入内容
        self.input_Q_init_1 = [float(x) for x in input_text.split(',')]  # 将输入内容解析为数组

        # print("User input array:", input_Q_init)

        # 获取模型的超参数设置
        self.alpha_1 = self.alpha_doubleSpinBox_2.value()  # 获取用户设置的学习率
        self.alpha_init_1 = self.alpha_init_Edit_2.text()  
        self.gramma_1 = self.grmma_doubleSpinBox_2.value()  
        self.theta_1 = self.theta_doubleSpinBox_2.value() 
        self.theta_p_1 = self.theta_p_doubleSpinBox_2.value() 
        self.pe_init_1 = self.PElineEdit_2.text()

        # print(alpha,alpha_init,gramma,theta,theta_p,pe_init)
        # return input_Q_init
    

    # 定义 Q 值迭代过程的函数
    def get_Q_iterm_value_1(self):
        '''
        df: dataframe 传入数据行为每一次试验,column0:每次试验所给的信号,column1:为每次输出的结果或者动作.
        N_STATES: 所切换的状态的总数,如试验的总次数.
        ALPHA: float 取值为0~1,学习率
        GAMMA: float 取值为0~1,折扣率
        算法原理（贝尔曼方程）
        Q(s, a) = Q(s, a) + α(r + γ * Q(s', a') - Q(s, a))
        '''
        self.Q_init_input_1()
        # 迭代超参数设置
        THETA = self.theta_1
        THETA_p = self.theta_p_1
        PE = self.pe_init_1
        ALPHA = self.alpha_1
        GAMMA = self.gramma_1
        ALPHA_init = self.alpha_init_1

        Q_go = []                         # 创建存储迭代过程的go_Q值空列表
        alpha_list = []                   # 创建存储迭代过程的alpha值空列表
        pre_err = []                      # 创建存储迭代过程的PE值空列表

        action_Q_list = self.input_Q_init_1

        Q_go.append(list(action_Q_list))
        alpha_list.append(ALPHA)
        pre_err.append(PE)

        go_N_STATES = np.array(self.dataFrame_1).shape[0]

        for i in range(go_N_STATES-1):   
            
            R,ActionSig = self.dataFrame_1.loc[i,'reward'],self.dataFrame_1.loc[i,'action_type']
            next_R,next_ActionSig = self.dataFrame_1.loc[i+1,'reward'],self.dataFrame_1.loc[i+1,'action_type']
            if self.get_Q_model_type_2() == 'Auto-alpha-TD(0)':
                # Auto_alpha_TD（0）算法更新方式
                PE= R - action_Q_list[ActionSig-1]
                ALPHA = THETA_p*ALPHA+THETA*abs(PE)
                action_Q_list[ActionSig-1] = action_Q_list[ActionSig-1] + ALPHA*(R - action_Q_list[ActionSig-1])
                # print(action_Q_list)
            elif self.get_Q_model_type_2() == 'SARSA':
                action_Q_list[ActionSig-1] = action_Q_list[ActionSig-1] + ALPHA*(R + GAMMA*next_R - action_Q_list[ActionSig-1])
            else:
                QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
                pass 

            Q_go.append(list(action_Q_list))
            alpha_list.append(ALPHA)
            pre_err.append(PE)

        # self.Q_go_array = np.array(Q_go)
        self.Q_go_array = Q_go

        Q_action_df = pd.DataFrame(self.Q_go_array, columns=[f'Qaciton{m}' for m in range(len(action_Q_list))])  # Convert to DataFrame

        dff_1 = pd.concat([self.dataFrame_1,Q_action_df],axis=1)
        # 增加动态的学习率和PE进去
        dff_1['Alpha'] = alpha_list
        dff_1['PE'] = pre_err
        self.Q_dff_1 = dff_1
        # print(self.Q_dff_1)
        if self.get_P_model_type_2() == 'Softmax':
            P = softmax_update(self.Q_go_array,self.doubleSpinBox_3.value())
        elif self.get_P_model_type_2() == 'Sigmoid':
            P = sigmoid_update(self.Q_go_array,self.doubleSpinBox_3.value())
        else:
            QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
            pass 
        prob = pd.DataFrame(P, columns=[f'Paciton{n}' for n in range(len(self.input_Q_init_1))])
        self.QP_df_1 = pd.concat([self.Q_dff_1,prob],axis=1)
        print(self.QP_df_1)


    # 定义依概率随机生成决策函数
    def generate_action(self,init_Q_list,len_num,Scaling_factor,pre_sig):
        '''
        更新迭代公式后的动作序列生成函数
        init_Q_list: 起始Q值序列
        action_len: 模拟生成的行动序列长度
        pre_sig:预测标志位，1：代表预测模型，0代表仿真模型
        '''
        N = len_num    # len
        if pre_sig == 1:
            # Learning rate and discount factor
            THETA = self.theta_1
            THETA_p = self.theta_p_1
            PE = self.pe_init_1
            ALPHA = self.alpha_1
            GAMMA = self.gramma_1
            ALPHA_init = self.alpha_init_1
        else:
            THETA = self.theta_2
            THETA_p = self.theta_p_2
            PE = self.pe_init_2
            ALPHA = self.alpha_2
            GAMMA = self.gramma_2
            ALPHA_init = self.alpha_init_2

        Q_name_list = [f'Qaciton{m}' for m in range(len(init_Q_list))]
        P_name_list = [f'Paciton{n}' for n in range(len(init_Q_list))]
        self.Q_name_list = Q_name_list
        # Create an empty DataFrame to record data
        columns = Q_name_list+P_name_list+['outcome','action_type', 'Alpha', 'PE']
        data = pd.DataFrame(columns=columns)
        # print(data)
        # Iterate to generate data
        P_model_type_2 = self.get_P_model_type_2()
        P_model_type_3 = self.get_P_model_type_3()
        Q_model_type_2 = self.get_Q_model_type_2()
        Q_model_type_3 = self.get_Q_model_type_2()


        for i in range(N):
            Q = np.array([init_Q_list])
            if pre_sig == 1:
                if P_model_type_2 == 'Softmax':
                    probs_list = softmax_update(Q,Scaling_factor)
                elif P_model_type_2 == 'Sigmoid':
                    probs_list = sigmoid_update(Q,Scaling_factor)
                else:
                    QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
                    break
            else:
                if P_model_type_3 == 'Softmax':
                    probs_list = softmax_update(Q,Scaling_factor)
                elif P_model_type_3 == 'Sigmoid':
                    probs_list = sigmoid_update(Q,Scaling_factor)
                else:
                    QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
                    break
            # print(probs_list)
            # Choose action based on probabilities
            outcome = np.random.choice([z+1 for z in range(len(init_Q_list))], p=probs_list[0])
            # 当前动作的奖励
            reward = self.mapping[outcome]
            action_type = self.action_mapping[outcome]
                
            # Record data
            data.loc[i] = list(init_Q_list) + [float(x) for x in probs_list[0]] + [int(outcome),action_type, ALPHA, PE]

            # Update Q-values using learning rate and discount factor
            if pre_sig == 1:
                if Q_model_type_2 == 'Auto-alpha-TD(0)':
                    PE= reward - init_Q_list[outcome-1]
                    ALPHA = THETA_p*ALPHA+THETA*abs(PE)
                    init_Q_list[outcome-1] = init_Q_list[outcome-1] + ALPHA * (reward - init_Q_list[outcome-1])
                    init_Q_list = list(init_Q_list)
                elif Q_model_type_2 == 'SARSA':
                    init_Q_list[outcome-1] = init_Q_list[outcome-1] + ALPHA * (reward +GAMMA*reward - init_Q_list[outcome-1])
                    init_Q_list = list(init_Q_list)
                else:
                    QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
                    break
            else:
                if Q_model_type_3 == 'Auto-alpha-TD(0)':
                    PE= reward - init_Q_list[outcome-1]
                    ALPHA = THETA_p*ALPHA+THETA*abs(PE)
                    init_Q_list[outcome-1] = init_Q_list[outcome-1] + ALPHA * (reward - init_Q_list[outcome-1])
                    init_Q_list = list(init_Q_list)
                elif Q_model_type_3 == 'SARSA':
                    init_Q_list[outcome-1] = init_Q_list[outcome-1] + ALPHA * (reward +GAMMA*reward - init_Q_list[outcome-1])
                    init_Q_list = list(init_Q_list)
                else:
                    QtWidgets.QMessageBox.information(self, 'User define', '用户根据需求修改底层代码') 
                    break
                
        return data
    

    # 定义预测生成功能函数
    def predict_function(self):
        
        Q_name_list = [f'Qaciton{m}' for m in range(len(self.input_Q_init_1))]
        Q_list_init = list(self.QP_df_1.loc[self.QP_df_1.index[-1],Q_name_list])
        # print(Q_list_init)
        self.pre_df = self.generate_action(Q_list_init,int(self.spinBox_2.value()),self.doubleSpinBox_3.value(),1)
        # print(pre_df)
        pre_and_ori_df = pd.concat([self.QP_df_1,self.pre_df],axis=0)
        pre_and_ori_df.reset_index(inplace=True,drop=True)
        self.pre_and_ori_df = pre_and_ori_df
        # print(self.pre_and_ori_df)
        self.pre_plot_data()
        self.pre_plot_data_P()

    def save_data_1(self):
        if self.pre_and_ori_df is not None:  # 替换需要最终要保存的dataframe
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save CSV File', '', 'CSV Files (*.csv)')
            if save_path:
                self.pre_and_ori_df.to_csv(save_path, index=False)
                QtWidgets.QMessageBox.information(self, 'Save Successful', 'DataFrame saved successfully.')


        # 预测数据的QP绘图函数
    def pre_plot_data(self):
        # df = self.pre_df
        df = self.pre_and_ori_df
        self.figure3.clear()
        ax = self.figure3.add_subplot(111)
        ax.plot(df.iloc[:len(self.QP_df_1),:].index,df.iloc[:len(self.QP_df_1),:]['Qaciton0'], marker='o',c='#C17F6F')
        ax.plot(df.iloc[len(self.QP_df_1):,:].index,df.iloc[len(self.QP_df_1):,:]['Qaciton0'], marker='o',c='#E6B6A8')

        ax.set_xlabel('trail by trail')
        ax.set_ylabel('Q_value')
        ax.set_title('Q—vari_trail by trail')
        self.canvas3.draw()


    def pre_plot_data_P(self):
        # df = self.pre_df
        df = self.pre_and_ori_df
        self.figure4.clear()
        ax = self.figure4.add_subplot(111)
        # ax.plot(df.index, df['Paciton0'], marker='o')
        ax.plot(df.iloc[:len(self.QP_df_1),:].index,df.iloc[:len(self.QP_df_1),:]['Paciton0'], marker='o',c='#C17F6F')
        ax.plot(df.iloc[len(self.QP_df_1):,:].index,df.iloc[len(self.QP_df_1):,:]['Paciton0'], marker='o',c='#E6B6A8')
        ax.set_xlabel('trail by trail')
        ax.set_ylabel('P_value')
        ax.set_title('P—vari_trail by trail')
        self.canvas4.draw()


        # 将绘图结果保存到本地的功能函数
    def save_QP_fig(self):
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save fig File', '', 'jpg Files (*.jpg)')
            if save_path:
                self.figure3.savefig1(save_path)
                QtWidgets.QMessageBox.information(self, 'Save Successful', 'figure saved successfully.')
            save_path_1, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save fig File', '', 'jpg Files (*.jpg)')
            if save_path_1:
                self.figure4.savefig1(save_path_1)
                QtWidgets.QMessageBox.information(self, 'Save Successful', 'figure saved successfully.')


    # ======================================定义界面三的函数================================================
    # 定义读取数据
    def load_data_2(self):
        file_path,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'load File', '','Files (*.csv)') 

        if file_path:
            self.dataFrame_2 = pd.read_csv(file_path)
            # print(self.dataFrame)
            QtWidgets.QMessageBox.information(self, 'Read Successful', 'File read successfully.')

    # 定义奖励以及决策分类映射函数
    def display_dataframe_2(self, df):
        model = PandasModel(df)
        self.tableView_3.setModel(model)

    def display_mapping_2(self):
        self.mappingTable_3.setRowCount(len(self.dataFrame_2['outcome'].unique()))
        self.mappingTable_3.setColumnCount(3)
        self.mappingTable_3.setHorizontalHeaderLabels(['outcome', 'reward','action_type'])

        for idx, value in enumerate(self.dataFrame_2['outcome'].unique()):
            value_item = QtWidgets.QTableWidgetItem(str(value))

            self.mappingTable_3.setItem(idx, 0, value_item)

            if value in self.mapping:
                mapping_value = self.mapping[value]
                mapping_item = QtWidgets.QTableWidgetItem(str(mapping_value))
                self.mappingTable_3.setItem(idx, 1, mapping_item)

    def save_mapping_2(self):
        for row in range(self.mappingTable_3.rowCount()):
            value_item = self.mappingTable_3.item(row, 0)
            mapping_item = self.mappingTable_3.item(row, 1)
            mapping_item2 = self.mappingTable_3.item(row, 2)

            if value_item and mapping_item:
                value = value_item.text()
                mapping_value = float(mapping_item.text())
                mapping_value_2 = int(mapping_item2.text())  # 获取动作类别标签

                self.mapping[int(value)] = mapping_value
                self.action_mapping[int(value)] = mapping_value_2

        if  self.dataFrame_2 is not None:
            self.dataFrame_2['reward'] = self.dataFrame_2['outcome'].map(self.mapping)
            self.dataFrame_2['action_type'] = self.dataFrame_2['outcome'].map(self.action_mapping)
            # 显示带有 reward 列的 DataFrame
            # self.display_dataframe_1(self.dataFrame_1)
            QtWidgets.QMessageBox.information(self, 'Set Successful','Set map Successful')

        return self.dataFrame_2
    

        # 获取用户设置的不同动作的迭代初值以及模型超参数
    def Q_init_input_2(self):
        input_text = self.Qinit_LineEdit_3.text()  # 获取文本框中的输入内容
        self.input_Q_init_2 = [float(x) for x in input_text.split(',')]  # 将输入内容解析为数组

        # print("User input array:", input_Q_init)

        # 获取模型的超参数设置
        self.alpha_2 = self.alpha_doubleSpinBox_3.value()  # 获取用户设置的学习率
        self.alpha_init_2 = self.alpha_init_Edit_3.text()  
        self.gramma_2 = self.grmma_doubleSpinBox_3.value()  
        self.theta_2 = self.theta_doubleSpinBox_3.value() 
        self.theta_p_2 = self.theta_p_doubleSpinBox_3.value() 
        self.pe_init_2 = self.PElineEdit_3.text()

    
    # 定义仿真生成函数
    def generate_function(self):
        self.Q_init_input_2()
        Q_list_init = list(self.input_Q_init_2)
        # print(Q_list_init)
        G_df = pd.DataFrame()
        for k in range(self.spinBox.value()):
            gen_df = self.generate_action(Q_list_init,int(self.spinBox_3.value()),self.doubleSpinBox_6.value(),0)
            gen_df['iter_num'] = k
            print(gen_df)
            G_df = pd.concat([G_df,gen_df],axis=0)

        self.G_df = G_df
        print(G_df)

    # 生成数据的预览函数
    def display_generate_df(self):
        self.display_dataframe_2(self.G_df)
        self.gen_plot_data()
        self.gen_plot_data_P()

    # 生成数据的QP绘图函数
    def gen_plot_data(self):
        df = self.G_df[self.G_df['iter_num']==0]
        self.figure5.clear()
        ax = self.figure5.add_subplot(111)
        ax.plot(df.index, df['Qaciton0'], marker='o', c='#C17F6F')
        ax.set_xlabel('trail by trail')
        ax.set_ylabel('Q_value')
        ax.set_title('Q—vari_trail by trail')
        self.canvas5.draw()

    def gen_plot_data_P(self):
        df = self.G_df[self.G_df['iter_num']==0]
        self.figure6.clear()
        ax = self.figure6.add_subplot(111)
        ax.plot(df.index, df['Paciton0'], marker='o', c='#C17F6F')
        ax.set_xlabel('trail by trail')
        ax.set_ylabel('P_value')
        ax.set_title('P—vari_trail by trail')
        self.canvas6.draw()


    def save_data_2(self):
        if self.G_df is not None:  # 替换需要最终要保存的dataframe
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save CSV File', '', 'CSV Files (*.csv)')
            if save_path:
                self.G_df.to_csv(save_path, index=False)
                QtWidgets.QMessageBox.information(self, 'Save Successful', 'DataFrame saved successfully.')


    # ===================================主功能菜单页面切换函数=================================================
    def set_page_1(self):
        self.stackedWidget.setCurrentIndex(0)

    def set_page_2(self):
        self.stackedWidget.setCurrentIndex(1)

    def set_page_3(self):
        self.stackedWidget.setCurrentIndex(2)
            
    def set_page_4(self):
        self.stackedWidget.setCurrentIndex(3)

    def set_page_5(self):
        self.stackedWidget.setCurrentIndex(4)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = DataVisualizationApp()
    window.show()
    sys.exit(app.exec_())




# 显示进度条设置
        # for i in range (100):
        #     time.sleep(0.5)
        #     self.progressBar.setValue(i)

# SARSA 算法的更新方式
    # if lickSig ==1:
    #     Qs_go_lick = Qs_go_lick + ALPHA*(R + GAMMA * Qs_go_next_R - Qs_go_lick)
    # else:
    #     Qs_go_nolick = Qs_go_nolick + ALPHA*(R + GAMMA * Qs_go_next_R - Qs_go_nolick)