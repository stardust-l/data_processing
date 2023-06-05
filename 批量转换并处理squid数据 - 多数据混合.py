from email import header
import matplotlib.pyplot as plt
import numpy as np
import os,sys
from PIL import Image

'''切换到当前文件所在目录'''
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)
import common
def sep_data(data,if_show=True,temp=None):
    #根据温度拆分数据
    dat_list=[]
    if type(None)==type(temp):
        temp=np.sort(list(set(np.round(data[0]))) ,axis=None)
    for i in range(len(temp)):
        T=temp[i]
        dat=[[data[1][j],data[2][j]] for j in range(len(data[0])) if round(data[0][j])==T]
        dat_list.append(np.transpose(dat)) 

    common.plot_mul_fig(data=dat_list,legend=[str(s) for s in temp],ylabel='$R_{xy}$')
    return dat_list,temp
def get_M(data,temp,fold_save='',name=None):
    RH_list=[]
    data_list=[]
    for i in range(len(data)):
        dat=data[i]
        RH,dat_i=common.get_height(dat,height_type=1,if_plot=1,return_data=True)
        RH_list.append(RH)
        data_list.append(dat_i)
        if(fold_save):
            np.savetxt(fold_save+'/'+name[:-4]+'T_%.0fK_new.txt'%temp[i],np.transpose(dat_i),header='T%.0fK'%temp[i])
    common.plot_mul_fig(data=data_list,legend=[str(s) for s in temp],ylabel='$R_{xy}$')
    plt.plot(temp,RH_list,'ko')#300,275,250,225
    print(temp,RH_list)
    plt.show()
    return RH_list
def main(key_words_list,key_common,delta_R=True,base_R=0,fold='./',if_save=False,fold_save=None,bridge=1,temp=None):
    if type(fold_save)==type(None):
        fold_save=fold+'\data_new'
        if not os.path.exists(fold_save):
            #不存在输出文件夹时创建
            os.makedirs(fold_save)
    #预处理函数
    data=[]
    ##读取数据以及处理

    head=''

    min_R=0
    for key_words in key_words_list:
        #寻找文件，导入数据
        f=common.find_file([head+key_words]+key_common,fold,file_type='dat')[0]
        name=f

        dat=np.loadtxt(fold+'/'+name,unpack=True,usecols=(2,3,4),skiprows=36,delimiter=',')

        dat_list,temp=sep_data(dat,temp=temp)
        get_M(dat_list,temp,fold_save=fold_save,name=f)
        dat=dat[1:]
        #dat=common.remove_data_LF(dat,H_limt=[-105,105])#
        #dat=common.remove_line_background(dat,x_start=50000)
        #dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=0)#纵轴移动到中点
        #height=-common.get_height(dat)-1
        dat[1]+=0#min_R+height
        
        if(if_save and 0):
            np.savetxt(fold_save+'/'+name[:-4]+'_new.txt'%bridge,np.transpose(dat),header=key_words)
        
        
        data.append(dat)

    print(key_words_list)

    common.plot_mul_fig(data=data,legend=key_words_list,ylabel='$R_{xy}$')
if __name__=='__main__':
    key_words_list=['300_10K']
    key_words_common=['0406','2p48']
    temp=[300,200,175][:]
    fold=r'D:\OneDrive - tju.edu.cn\科研\CrPt\data\SQUID-VSM-M03\202304'
    #fold_save=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\霍尔回线\data_new'
    main(key_words_list=key_words_list,key_common=key_words_common,fold=fold,if_save=True,bridge=3,temp=temp)
    plt.show()