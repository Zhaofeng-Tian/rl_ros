
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
 
def plot1():
    name_list = ['Monday','Tuesday','Friday','Sunday']
    num_list = [1.5,0.6,7.8,6]
    num_list1 = [1,2,3,1]
    x =list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    
    plt.bar(x, num_list, width=width, label='boy',fc = 'y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='girl',tick_label = name_list,fc = 'r')
    plt.legend()
    plt.show()


def plot2(): 
    name_list = ['SR','FIRect','FIFR']
    num_list = [508,407,363]
    colors = [[254/255,198/255,114/255],[168/255,183/255,200/255], [186/255,217/255,151/255]]
    plt.bar(range(len(num_list)), num_list,color=colors,edgecolor = 'black' ,width = 0.5,tick_label=name_list)
    font2 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 30, }
    # plt.xlabel('Representations', font2)
    plt.xticks(fontproperties = 'Times New Roman', size=20)
    plt.yticks(fontproperties = 'Times New Roman',size=20)

    plt.show()

if __name__ == "__main__":
    plot2()