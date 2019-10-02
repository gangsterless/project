import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def PlotBar(name,X,Y):#画条形图
    """
    绘制水平条形图方法barh
    参数一：y轴
    参数二：x轴
    """
    plt.barh(range(len(X)), Y, height=0.7, color='steelblue', alpha=0.8)  # 从下往上画
    plt.yticks(range(len(X)), X)
    # plt.xlim(30, 47)
    plt.title(name)
    for x, y in enumerate(Y):
        plt.text(y + 0.2, x - 0.1, '%s' % y)
    plt.show()
