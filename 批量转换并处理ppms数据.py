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
def main(key_words_list,key_common,delta_R=True,base_R=0,fold='./',if_save=False,fold_save=None,bridge=1):
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

        #磁场4，通道1-3电阻，6,8,10
        bridge_number=bridge#通道数
        dat=np.loadtxt(fold+'/'+name,unpack=True,usecols=(4,bridge_number*2+4),skiprows=33,delimiter=',')
        dat[0]=-dat[0]

        dat=common.select_data(dat,n=0)#
        dat=common.remove_odd_function(dat)

        dat=common.remove_line_background(dat,x_start=65000)
        dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=0)#纵轴移动到中点
        height=-common.get_height(dat)-1
        dat[1]+=0#min_R+height

        min_R=min(dat[1])
        if(if_save):
            np.savetxt(fold_save+'/'+name[:-4]+'_bridge%g-_new.txt'%bridge,np.transpose(dat),header=key_words)

        
        data.append(dat)

    print(key_words_list)

    common.plot_mul_fig(data=data,legend=key_words_list,ylabel='$R_{xy}$')
if __name__=='__main__':
    key_words_list=['2p1']
    key_words_common=['20230412','0424','800C_5h-AHE-300K']
    fold=r'D:\OneDrive - tju.edu.cn\科研\CrPt\data\ppms\2023\202305'
    #fold_save=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\霍尔回线\data_new'
    main(key_words_list=key_words_list,key_common=key_words_common,fold=fold,if_save=True,bridge=1)
    plt.show()