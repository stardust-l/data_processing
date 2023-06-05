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
    ##读取数据以及处理

    head=''

    min_R=0
    for key_words in key_words_list:
        #寻找文件，导入数据
        f=common.find_file([head+key_words]+key_common,fold)[0]
        name=f

        #去除头两个点
        dat=np.loadtxt(fold+'/'+name,unpack=True)
        dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=0)#纵轴移动到中点

        height=-common.get_height(dat)-1
        dat[1]+=0#min_R+height
        min_R=min(dat[1])

        np.savetxt(fold_save+'/'+name[:-4]+'_new.txt',np.transpose(dat),header=key_words)

        
        data.append(dat)

    print(key_words_list)

    common.plot_mul_fig(data=data,legend=key_words_list)
if __name__=='__main__':
    key_words_list=['Ta30']
    key_words_common=['CFB10','PHE','1°']
    fold=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\V-Mo-SOT\202211变CFB厚度\20230105样品\Ta30%Mo10'
    #fold_save=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\霍尔回线\data_new'
    main(key_words_list=key_words_list,key_common=key_words_common,fold=fold)
    plt.show()