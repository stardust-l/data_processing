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
    if max(abs(delta_list))>abs(np.median(delta_list))*20:
        if(data[1][0]<max(data[1])):
            data_new=data[:,:index_change_fast-1]
        else:
            data_new=data[:,index_change_fast+1:]
    else:
        data_new=data[:,:]
    if(if_print):
        print('raw',data[:,:5])
        print('res:',data_new[:,:5])
    return data_new

def harm1(data,H_limt=None,if_show=True,if_save=False,fold='./',save_limt=None):
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
    data2_new=remove_data_LF(data2,H_limt=[-H_limt[-1],-H_limt[0]]) 
    data1_new[1]=data1_new[1]-np.median(data1_new[1])
    data2_new[1]=data2_new[1]-np.median(data2_new[1])
    #去除翻转后的数据
    #data1_new=remove_switch(data1_new)
    #data2_new=remove_switch(data2_new)
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
    if(if_save):
        import re
        x_fit=np.linspace(save_limt[0],save_limt[1],50)
        data_save=[*data1_new,x_fit,np.polyval(params1,x_fit),*data2_new,x_fit,np.polyval(params2,x_fit)]
        import pandas as pd
        name='_harm1.txt'
        df=pd.DataFrame(data_save,index=['H_+M','Vy_+M','H_fit','V_fit_+M','H_-M','Vy_-M','H_fit','V_fit_-M'])
        df=df.T
        df.to_csv(fold+name,index=None)
        #np.savetxt(fold+name,data_save)
    return np.array([params1[0],params2[0]])
def harm2(data,H1_limt=None,H2_limt=None,if_show=True,if_select=True,if_save=False,fold='./',save_limt=None):
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
    data2_new=remove_data_LF(data2,H_limt=[-H1_limt[-1],-H1_limt[0]])
    #去除翻转后的数据
    #data1_new=remove_switch(data1_new)
    #data2_new=remove_switch(data2_new)
    data1_new[1]=data1_new[1]-np.median(data1_new[1])+0.05e-6
    data2_new[1]=data2_new[1]-np.median(data2_new[1])-0.05e-6
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
    if(if_save):
        import re
        x_fit=np.linspace(save_limt[0],save_limt[1],50)
        data_save=[*data1_new,x_fit,np.polyval(params1,x_fit),*data2_new,x_fit,np.polyval(params2,x_fit)]
        name='_harm2.txt'
        import pandas as pd
        df=pd.DataFrame(data_save,['H_+M','Vy_+M','H_fit','V_fit_+M','H_-M','Vy_-M','H_fit','V_fit_-M'])
        df=df.T
        df.to_csv(fold+name,index=None)
        #np.savetxt(fold+name,data_save)
    return np.array([params1[0],params2[0]])
def fit_harm_all(key_common,key_custom,fold='./',out_file='fit_harm2.txt',data_fold='./'):
    res=[]
    for key in key_custom:
        key_use=['%.1fmA'%key]
        file=common.find_file_walk(key_common+key_use,fold)[0]
        data=np.loadtxt(fold+'/'+file,unpack=True,usecols=[0,1,3])
        list_H=[-400,450]
        save_limt=[-500,500]
        
        params1=harm1(data[[0,1]],H_limt=list_H,if_save=True,fold=data_fold,save_limt=save_limt)
        print('一次谐波，二次项系数:',params1)
        params2=params1
        params2=harm2(data[[0,2]],H1_limt=list_H,if_select=False,if_save=True,fold=data_fold,save_limt=save_limt)
        print('二次谐波，一次项系数:',params2)
        H_eff=-2*params2/params1
        print('有效场',np.mean(abs(H_eff)),'原始数据',H_eff)
        res.append([key,*H_eff,params1[0],params2[0]])
    print(res)
    #np.savetxt(out_file,res,header='I_eff  有效场 二次  一次')
fold=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\V-Mo-SOT\微加工\20221228-V30Mo10%CFB10%MgO30\B1\谐波2'

key_common=['iT']
keyword_cumtom=[4]
out_file=fold+'/'+'V30Mo10_BL'+'_%gmA'%keyword_cumtom[0]
fit_harm_all(key_common=key_common,key_custom=keyword_cumtom,fold=fold,data_fold=out_file)


