import matplotlib.pyplot as plt
import os,common,sys
import numpy as np

current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)

#分流效应计算
rho_CFB=105
rho_W=140
t_W=4e-7
t_CFB=1e-7
w_line=20e-4
UNIT=1e6#以MA/cm^2为单位
fenliu_div=w_line*t_W*UNIT/(rho_CFB*t_W)*(rho_CFB*t_W+rho_W*t_CFB) #由于分流效应，

fold=r'D:\OneDrive - tju.edu.cn\科研\轻金属的翻转\data\VSM\V-Mo-SOT\202212\20221228'
key=['6p248emu','-oop.V']
name=common.find_file_walk(key,fold=fold,file_type='VHD')[0]
dat=common.get_data(fold+'/'+name)#np.loadtxt(name,unpack=True)#fold+'/'+name+'.txt'
#dat=common.remove_line_background(dat,x_start=6900)
M=common.get_height(dat)#/0.8e-7/1*1000
t_CFB=0.8e-9
print(M)
print(fenliu_div*1000)
beta_DL=14*fenliu_div*1000/1e10/10000


import scipy.constants as sc
xi=2*sc.e/sc.hbar*M*t_CFB*beta_DL
print(xi)

plt.plot(*dat)
plt.show()