import matplotlib.pyplot as plt
import os,common,sys
import numpy as np
from numpy.core.defchararray import center
from brokenaxes import brokenaxes
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)

def main(fold,keyword_custom,keyword_common,title='',fold_save=None,path_background='',path_back_SF='',if_remove_back_all=0,if_save_Ms=False):
    if type(fold_save)==type(None):
        fold_save=fold+'\data_new'
        if not os.path.exists(fold_save):
            #不存在输出文件夹时创建
            os.makedirs(fold_save)
    files_list=os.listdir(fold)#找到要求目录下的所有文件
    name_list=[f.replace('.VHD','.VHD') for f in files_list if 'VHD' in f]
    if len(keyword_custom)==0:
        keyword_custom=[ ['%s'%i[:-4]] for i in name_list]

    data=[]
    Ms_data=[]
    data_header=[]
    if True:
        legend=[]
        
        for key in keyword_custom:
            #寻找文件，并载入数据
            print(key+keyword_common)
            name=common.find_file(key+keyword_common,fold=fold,file_type='VHD')[-1]
            dat=common.get_data(fold+'/'+name)
            if if_save_Ms:
                data_header.append(int(key[0][3:]))
            try:
                #去除杆的信号，并且去除大场时的
                if(dat[0][0]<-6000 or if_remove_back_all):
                    dat_back=common.get_data(path_background)
                    dat=common.remove_background_by_point(dat_back,dat)#去除杆背底
                    #dat=common.remove_skew(dat) #去除跳点
                    dat=common.remove_repeat_x(dat)
                    if(dat[0][0]<-7000):
                        dat=common.remove_line_background(dat,x_start=5000)
                    elif(dat[0][0]<-1000):
                        dat=common.remove_line_background(dat,x_start=2000)
                if(path_back_SF):
                    if(dat[0][0]>-200):
                        dat_back_SF=common.get_data(path_back_SF)
                        dat=common.remove_background_by_point(dat_back_SF,dat)#去除杆背底
                if(dat[0][0]>-200):
                    dat=common.remove_line_background(dat,x_start=45)
            except Exception as e:
                print(e,'\n找不到空杆信号的文件，继续下一步')
                pass

            dat=common.remove_skew(dat) #去除跳点
            #dat[1]/=0.8*1e-7*0.49#算磁化强度
            dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=0)
            data.append(dat)
            np.savetxt(fold_save+'/'+name[:-4]+'.txt',np.transpose(dat),header=key[0])

            Ms_temp=common.get_height(dat,height_type=0)
            Ms_data.append(Ms_temp)
        if if_save_Ms:
            name_Ms='-'.join(keyword_common+['Ms'])
            np.savetxt(fold_save+'/'+name_Ms+'.txt',np.transpose([data_header,Ms_data]))

    else:
        for name in name_list:
            dat=np.loadtxt(fold+'/'+name+'.txt',unpack=True)#fold+'/'+name+'.txt'
            dat=common.move_center_to_zero(dat,if_guiyi=0,center=0,tongxiang=1)
            data.append(dat)
        legend=[n.replace('.txt','')[4:] for n in name_list]
        linecolor=None#['k-','k:','k--','r','b']
    title=''

    legend=[key[0] for key in keyword_custom]
    common.plot_mul_fig(data,legend=legend,ylabel=r'$M_s\ (emu/cc)$',fig_name=title,show_grid=False,n_line_style=2)
    #dat=[]
    #for da in data:
    #   dat.extend(da)
    #np.savetxt(title+'.txt',np.transpose(dat),fmt='%g')

if __name__=='__main__':
    plt.rcParams['font.size']=14 #设置图像中所有字体的大小
    title=''#不为空则保存图 #'VSM-W(4)CoFeB(1)MgO(3)-outplane-20220516'
    keyword_custom=''#os.listdir()#[['CFB10']]
    keyword_common=['']#['Ta30%Mo10','300C','120min']

    fold=r'C:\Users\lgs\Documents\WeChat Files\wxid_l2tbxn558wy022\FileStorage\File\2023-02\he-20220916sample\he-20220916sample'
    path_back=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\VSM\V-Mo-SOT\RH-T\20230208-konggan-oop-7000Oe.VHD'
    path_back_SF=''#r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\VSM\V-Mo-SOT\202301\20230105\20230105-konggan-ip-100Oe.VHD'
    main(fold=fold,keyword_custom=keyword_custom,title=title,keyword_common=keyword_common,path_background=path_back,path_back_SF=path_back_SF,if_remove_back_all=0)