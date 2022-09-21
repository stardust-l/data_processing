
import numpy as np
import matplotlib.pyplot as plt
import os,struct,sys
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)

"""参数设置"""
path='Pt%Al%Pt.raw'
if_save_data=True#是否保存数据

"""读取数据"""
with open(path, 'rb') as f:
    data = f.read()
print(data[:7])
import bruker
data_new=bruker.loads(data)
#print(data_new)

"""数据处理"""
dat=data_new['data'][0]['values']['count']#获取光强数据
data_new=data_new['data'][0]
#print(data_new['two_theta_start'],data_new['increment_1'],data_new['no_of_measured_data'],len(dat))

#判断扫描方式 通过scan_type,1:2theta-omega扫描 3：theta扫描
if(data_new['scan_type']==1):
    #1:theta-2theta扫描
    data_start=data_new['two_theta_start']
elif(data_new['scan_type']==3):
    #3：theta扫描
    data_start=data_new['theta_start']
else:
    print('未设置的扫描方式，等待以后编写代码')
    sys.exit()
stop=data_start+data_new['increment_1']*(data_new['no_of_measured_data']-1)
x=np.linspace(data_start,stop,data_new['no_of_measured_data'])

"""绘图与数据保存"""
plt.semilogy(x,dat)
plt.show()
if if_save_data:
    with open(path[:-3]+'txt','w') as f:
        for dat in zip(x,dat):
            f.write(str(dat[0])+' '+str(dat[1])+'\n')
