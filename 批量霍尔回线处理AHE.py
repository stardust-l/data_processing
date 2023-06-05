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
def main(key_words_list,key_common,delta_R=True,base_R=0,fold='./',if_save=False,fold_save=None):
    if type(fold_save)==type(None):
        fold_save=fold+'\data_new'
        if not os.path.exists(fold_save):
            #不存在输出文件夹时创建
            os.makedirs(fold_save)
    #预处理函数
    data=[]
    RH_list=[]
    ##读取数据以及处理

    head=''

    min_R=0
    for key_words in key_words_list:
        #寻找文件，导入数据
        f=common.find_file([head+key_words]+key_common,fold)[0]
        name=f

        #去除头两个点
        dat=np.loadtxt(fold+'/'+name,unpack=True)
        dat[0]=dat[0]
        #dat=common.remove_data_LF(dat,H_limt=[-2.0e4,2.0e4])#
        #dat=common.remove_line_background(dat,x_start=1.6e4)
        dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=0)#纵轴移动到中点
        height=common.get_height(dat)
        #height_zero=common.get_height_zero(dat)
        #print('霍尔电阻大小为%.2f，零点大小为%.3f,方形度%.1f%%'%(height,height_zero,height_zero/height*100))
        dat[1]+=1.127e-6-1.10e-6#min_R+height
        
        min_R=min(dat[1])
        if(if_save):
            np.savetxt(fold_save+'/'+name[:-4]+'+new.txt',np.transpose(dat),header=key_words)

        RH_list.append(round(height,2))

        data.append(dat)

    print(key_words_list)
    print(RH_list)

    common.plot_mul_fig(data=data,legend=key_words_list)
if __name__=='__main__':
    key_words_list=['CrPt2p45'][:]
    key_words_common=['800C']
    fold=r'D:\OneDrive - tju.edu.cn\科研\CrPt\data\电输运\CrPt3\摸条件\202304\20230418'
    #fold_save=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\霍尔回线\data_new'
    main(key_words_list=key_words_list,key_common=key_words_common,fold=fold,if_save=True)
    plt.show()