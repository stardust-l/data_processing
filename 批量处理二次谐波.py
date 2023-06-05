import select
from struct import unpack
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from numpy.core.defchararray import center
from scipy import interpolate
from scipy.interpolate import interp1d
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)
sys.path.append('../')
import common
#SMR xlabel=r'$\theta\ (\degree)$',ylabel=r'$R_{xy}\ (\Omega)$'
#AHE xlabel=r'$B\ (Gs)$',ylabel=r'$R_{xy}\ (\Omega)$'
#switch xlabel=r'$I_x\ (A)$',ylabel=r'$R_{xy}\ (\Omega)$'


#
def remove_data_LF(data,H_limt=None,H_max=None):
    #取小场下数据
    data_new=np.transpose([dat for dat in data if dat[0]>H_limt[0] and dat[0]<H_limt[1]])
    return data_new
def remove_pianlian(data,error=5,n=1,k0=None):
    #去除偏离过大的点
    if(type(k0)!=type(None)):
        params=[0,np.mean(data[1])]
    else:
        params=np.polyfit(*data, n) 
    while(True):
        dat=[[],[]]
        delta=[abs(da[1]-np.polyval(params,da[0])) for da in np.transpose(data)]
        mean=np.median(delta)
        error_index=[]
        for i in range(len(delta)):
            if abs(data[1][i]-np.polyval(params,data[0][i]))>error*mean:
                error_index.append([i,delta[i],mean])
            else:
                dat[0].append(data[0][i])
                dat[1].append(data[1][i])
        data=dat
        #print(data)
        params=np.polyfit(*data, n) 
        if(len(error_index)==0):
            return np.array(data),params
def remove_switch(data,if_print=False):
    #去除翻转后的数据
    #间隔均值
    delta_list=data[1][:-1]-data[1][1:]
    index_change_fast=np.argmax(np.abs(delta_list))#变化最快的点
    if max(abs(delta_list))>np.median(abs(delta_list))*10:
        if(data[0][0]<data[0][-1]):
            data_new=data[:,:index_change_fast-1]
        else:
            data_new=data[:,:index_change_fast-1]
    else:
        data_new=data[:,:]
    if(if_print):
        print('raw',data[:,:5])
        print('res:',data_new[:,:5])
    return data_new

def harm1(data,H_limt=None,if_show=True):
    #二次谐波，小场下二次函数拟合
    #数据分成两段
    index_max_data=np.argmax(data[0])
    data1=np.transpose(data[:,:index_max_data])
    data2=np.transpose(data[:,index_max_data:])
    #先看一下数据
    plt.plot(*np.transpose(data1),'ko')
    plt.plot(*np.transpose(data2),'ro')
    plt.show()
    #给出新范围下的数据
    data1_new=remove_data_LF(data1,H_limt=H_limt)
    data2_new=remove_data_LF(data2,H_limt=H_limt) 
    data1_new[1]=data1_new[1]-np.median(data1_new[1])
    data2_new[1]=data2_new[1]-np.median(data2_new[1])
    #去除翻转后的数据
    data1_new=remove_switch(data1_new)
    data2_new=remove_switch(data2_new)
    #去除突变点
    data1_new,params1=remove_pianlian(data1_new,n=2,k0=0)
    data2_new,params2=remove_pianlian(data2_new,n=2,k0=0)  
    data1_new,params1=remove_pianlian(data1_new,n=2)
    data2_new,params2=remove_pianlian(data2_new,n=2)  
    #params1=np.polyfit(*data1_new, 2)
    #params2=np.polyfit(*data2_new, 2)
    if(if_show):
        plt.plot(*data1_new,'ko',label='data1')
        plt.plot(data1_new[0],np.polyval(params1,data1_new[0]),'k-',label='fit1')
        plt.plot(*data2_new,'go',label='data2')
        plt.plot(data2_new[0],np.polyval(params2,data2_new[0]),'g-',label='fit2')
        plt.legend()
        plt.show()
    return np.array([params1[0],params2[0]])
def harm2(data,H1_limt=None,H2_limt=None,if_show=True,if_select=True):
    #二次谐波，小场下线性拟合
    #数据分成两段
    index_max_data=np.argmax(data[0])
    data1=np.transpose(data[:,:index_max_data])
    data2=np.transpose(data[:,index_max_data:])
    #先看一下数据
    plt.plot(*np.transpose(data1),'ko')
    plt.plot(*np.transpose(data2),'ro')
    plt.show()
    #给出新范围下的数据
    data1_new=remove_data_LF(data1,H_limt=H1_limt)
    data2_new=remove_data_LF(data2,H_limt=H2_limt)
    #去除翻转后的数据
    data1_new=remove_switch(data1_new)
    data2_new=remove_switch(data2_new)
    #while(True):
    #    #线性拟合，并去除偏离过大的点

    
    k1,b1=np.polyfit(*data1_new, 1) 
    plt.plot(*data1_new,'ro',label='data1_raw')
    plt.plot(data1_new[0],k1*data1_new[0]+b1,'r--',label='fit1_raw')
    data1_new,params1=remove_pianlian(data1_new)
    data2_new,params2=remove_pianlian(data2_new)

    if(if_show):
        plt.plot(*data1_new,'ko',label='data1')
        plt.plot(data1_new[0],np.polyval(params1,data1_new[0]),'k-',label='fit1')
        plt.plot(*data2_new,'go',label='data2')
        plt.plot(data2_new[0],np.polyval(params2,data2_new[0]),'g-',label='fit2')
        plt.legend()
        plt.show()
    return np.array([params1[0],params2[0]])
def fit_harm_all(key_common,key_custom,fold='./',out_file='fit_harm2.txt'):
    res=[]
    for key in key_custom:
        key_use=['{}mA'.format(key/1000*1000)]
        file=common.find_file_walk(key_common+key_use,fold)[0]
        data=np.loadtxt(fold+'/'+file,unpack=True,usecols=[0,1,3])
        H_limt=[-800,800]
        params1=harm1(data[[0,1]],H_limt=H_limt)
        print('二次:',params1)
        params2=harm2(data[[0,2]],H1_limt=H_limt,H2_limt=[-H_limt[1],-H_limt[0]],if_select=False)
        print('一次:',params2)
        H_eff=-2*params2/params1
        print('有效场',np.mean(abs(H_eff)),'原始数据',H_eff)
        res.append([key,*H_eff,params1[0],params2[0]])
    print(res)
    np.savetxt(out_file,res,header='I_eff  有效场 二次  一次')
fold=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\V-Mo-SOT\微加工\20221228-V30Mo10%CFB10%MgO30\B1\谐波2'

key_common=['iT','BL']
keyword_cumtom=[0.5*i for i in range(2,10)][:]
out_file=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\V-Mo-SOT\微加工\20221228-V30Mo10%CFB10%MgO30\B1\谐波2\BL.txt'
fit_harm_all(key_common=key_common,key_custom=keyword_cumtom,fold=fold,out_file=out_file)


