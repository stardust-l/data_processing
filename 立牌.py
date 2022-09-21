# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:08:39 2020

@author: win10
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2 as p2
import os

def print_seat(i,pdf_file=None):
    if not pdf_file:
        pdf_file='%i.pdf'%i
    s='%i-%i'%(i*10+1,i*10+10)
    
    d2=0.22
    d1=1.1/29.7
    d=d1+d2
    def plt_line(x,ax=None):
        if not ax:
            ax=plt.gca()
        ax.plot([0,1],[x,x],'k--',linewidth=0.5)
    def plt_h(x,ax=None):
        if not ax:
            ax=plt.gca()        
        ax.plot([x,x],[0,1],'k--',linewidth=0.5)
    #创建一个A4大小的画布
    fig=plt.figure(figsize=(8.27,11.75))
    #创建一个铺满整个画布的坐标轴,坐标轴范围为0,1
    ax=fig.add_axes([0, 0, 1, 1],xlim=(0,1),ylim=(0,1))
    #坐标轴不可见
    ax.set_axis_off()

    font = {'family' : 'kaiti',
            'weight' : 'bold',
            'size'   : '160'}
    plt.rc('font', **font)               # 步骤一（设置字体的更多属性）


    up=(d+3)/4
    down=(d*3+1)/4
    
    fig.text(0.5,up,s,horizontalalignment='center',verticalalignment='center',rotation=180)
    fig.text(0.5,down,s,horizontalalignment='center',verticalalignment='center',rotation=0)
    #plt_line(1)
    plt_line(d1)
    plt_line(d)
    plt_line(up*0.5+down*0.5)

    if type(pdf_file)==PdfPages:
        pdf_file.savefig(fig,dpi=300,papertype='a4')
    else:
        plt.savefig(pdf_file,dpi=300,papertype='a4')
    plt.close()
def print_attach(s,i=1,pdf_file=None,n=4):
    if not pdf_file:
        pdf_file='%i.pdf'%i
    #画水平线
    def plt_vline(x,ax=None):
        if not ax:
            ax=plt.gca()
        ax.plot([0,1],[x,x],'k--',linewidth=0.5,color='#101010')
    #画垂直线
    def plt_hline(x,ax=None):
        if not ax:
            ax=plt.gca()        
        ax.plot([x,x],[0,1],'k--',linewidth=0.5)
    def text(s,h,fig=None,theta=0):
        if not fig:
            fig=plt.gcf()
        fig.text(0.5,h,s,horizontalalignment='center',verticalalignment='center',rotation=theta,wrap=True)
    #创建一个A4大小的画布
    fig=plt.figure(figsize=(8.27,11.75))
    #创建一个铺满整个画布的坐标轴,坐标轴范围为0,1
    ax=fig.add_axes([0, 0, 1, 1],xlim=(0,1),ylim=(0,1))
    #坐标轴不可见
    ax.set_axis_off()
    #设置字体属性
    font = {'family' : 'kaiti',
            'weight' : 'bold',
            'size'   : '14'}
    plt.rc('font', **font)               

    for i in range(n):
        text(s,(i+0.5)/n)
        if i<n-1:
            plt_vline((i+1)/n)

    if type(pdf_file)==PdfPages:
        pdf_file.savefig(fig,dpi=300,papertype='a4')
    else:
        plt.savefig(pdf_file,dpi=300,papertype='a4')
    plt.close()
def creat2(num=25):
    pdf_merger = p2.PdfFileMerger(strict=True)
    for i in range(num):
        filename='%i.pdf'%i
        #生成一页要求的pdf
        print_seat(i,filename)
        #把pdf文件合并到一起
        with open(filename, "rb") as f:
            pdf_read=p2.PdfFileReader(f, 'rb')
            pdf_merger.append(pdf_read)
        #删除原始pdf
        os.remove(filename)
    pdf_merger.write('num.pdf')
    pdf_merger.close()
def creat1(num=25):
    with PdfPages('a.pdf','w') as pdf:
        for i in range(num):        
            #生成一页要求的pdf
            print_seat(i,pdf)
def f1():
    text='''联欢会制作人：曹昕宇
Q&A
Q：本次联欢会举办形式是什么？
A：本次是晚会and派对的全新形式，大家既可以观赏节目、参与集体游戏，也可以任意选择自己喜欢的游戏摊位玩耍～
Q：游戏礼券有什么用？
A：游戏礼券用于获取礼品，只有拥有礼券和编号才能获取游戏礼品。
Q：没有游戏礼券可以参加活动吗？
A：可以！游戏礼券只和奖品有关，没有礼券也可以参加任何想参加的活动！
Q：本次晚会有哪些内容？
A：本次晚会除了有多台舞蹈节目，还有集体参与的趣味问答和200人的超大型谁是卧底活动，最主要的就party time了～有桌游、娃娃机、switch等20余个游戏摊位供君选择～
Q：入场需要对号入座吗？
A：是的，但集体环节结束后，party time就可以想玩什么玩什么了～
Q：怎样获得奖品？
A：趣味问答（20份）、谁是卧底（20份）和娃娃机（活动大奖！）都可以获得优胜奖品，其中娃娃机的硬币可以通过玩其他游戏获得。
'''
    filename='attach.pdf'
    print_attach(text,pdf_file=filename,n=3)
if __name__=='__main__':
    creat2()
