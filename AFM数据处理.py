import numpy as np
import matplotlib.pyplot as plt
import os,struct,common
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)

def std(data):
    return np.sqrt(np.var(data))
def cal_rsm(data):
    rsm=np.sqrt(np.var(data))
    meadian=np.median(data)
    n_count=0
    sum_all=0
    sum_all_sq=0
    for i in range(len(data)):
        for j in range(len(data[0])):
            if 1 and abs(data[i][j]-meadian)<rsm*4:
                sum_all+=data[i][j]
                sum_all_sq+=data[i][j]**2
                n_count+=1
            else:
                data[i][j]=meadian
    rsm_new=np.sqrt(sum_all_sq/n_count-sum_all**2/n_count/n_count)
    return rsm_new
def get_rms(path,n=2):
    #把数据切割成多块，计算多块的rsm粗糙度
    data=np.loadtxt(path,unpack=True)
    plot_AFM(data)
    rsm_list=[]
    w,h=len(data)//n,len(data[0])//n
    for i in range(n):
        for j in range(n):
            data_temp=data[i*w:(i+1)*w,j*h:(j+1)*h]
            rsm=cal_rsm(data_temp)*1e9
            rsm_list.append(rsm)
    return rsm_list
def plot_AFM(data):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    #填色
    data_range=[0,4,0,4]#找出数据范围，以供画图时使用
    im1=ax.imshow(data,extent=data_range,cmap=None,origin="lower",interpolation='bilinear',aspect='auto')#extent=[-1,1,-1,1] , interpolation='nearest',
    
    #plt.colorbar()
    plt.show()
def main(key_words_list,key_common,fold='./',fold_save=None):
    if type(fold_save)==type(None):
        fold_save=fold+'\data_new'
        if not os.path.exists(fold_save):
            #不存在输出文件夹时创建
            os.makedirs(fold_save)
    #预处理函数
    data=[]
    ##读取数据以及处理

    for key_words in key_words_list:
        #寻找文件，导入数据
        f=common.find_file([key_words]+key_common,fold)[0]
        rsm=get_rms(fold+'/'+f,n=2)
        data.extend(rsm)
    rsm_end=common.cal_uAt(data)
    print(rsm_end)
'''计算RSM粗糙度'''

if __name__=='__main__':
    key_words_list=['good_1','good_2','good_3','good_4'][:3]
    key_words_common=['0319pn50_left']
    fold=r'D:\OneDrive - tju.edu.cn\科研\科研M07\数据\AFM\AFM-M03\20230328'
    #fold_save=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\霍尔回线\data_new'
    main(key_words_list=key_words_list,key_common=key_words_common,fold=fold)
