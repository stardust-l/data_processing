import matplotlib.pyplot as plt
import numpy as np
import os,sys
from matplotlib import ticker, cm
#将文件目录设定为工作目录
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)
'''零、参数设置,函数预定义'''
path='SC_399-sc_phase-fwd_xyz.txt'
def get_local_extremum_D2(data,n_judgment=2):
    #给定二维数据，返回二维极大值
    #n_judgment:需要连续大于附近的几个点，才认为它是极大值

    len_x=len(data[0])
    len_y=len(data)
    site_skyrmion=[]#存储找到的极大值的位置
    for i_y in range(len_y):
        for i_x in range(len_x):
            sum_judgment=0#中间变量，用于存储判断结果
            for i_judfment_x in range(-n_judgment,n_judgment+1):
                for i_judfment_y in range(-n_judgment,n_judgment+1):
                    if(i_x-i_judfment_x<0 or i_x-i_judfment_x>len_x-1 or i_y-i_judfment_y<0 or i_y-i_judfment_y>len_y-1):
                        #如果点超出边界，忽略这次判断
                        sum_judgment+=1
                    else:
                        sum_judgment+=(data[i_y][i_x]>=data[i_y-i_judfment_y][i_x-i_judfment_x])
            if(sum_judgment>=(n_judgment*2+1)**2):
                site_skyrmion.append([i_y,i_x])
    return site_skyrmion
def get_half_width(data,point,site):
    #找到指定点的半高宽，强度降到一半（均值和极大值的一半）的位置
    data_average=np.median(data.reshape(-1,1))
    data_half=(data_average+data[point[0]][point[1]])/2
    n_point_up_half=[0,0,0,0]#高于半值的点的距离
    n=0
    while(True):
        #右方向
        n+=1
        if(n+point[1]>len(data)-1):
            n_point_up_half[0]=0
            break
        if(data[point[0]][n+point[1]]>data_half):
            n_point_up_half[0]+=1
        else:
            y2=data[point[0]][n+point[1]]
            y1=data[point[0]][n-1+point[1]]
            n_point_up_half[0]+=(data_half-y1)/(y2-y1)
            break
    n=0
    while(True):
        #左方向
        n+=1
        if(-n+point[1]<0):
            n_point_up_half[1]=0
            break
        if(data[point[0]][-n+point[1]]>data_half):
            n_point_up_half[1]+=1
        else:
            y2=data[point[0]][-n+point[1]]
            y1=data[point[0]][-n+1+point[1]]
            n_point_up_half[1]+=(data_half-y1)/(y2-y1)
            break
    n=0
    while(True):
        #下方向
        n+=1
        if(-n+point[0]<0):
            n_point_up_half[2]=0
            break
        if(data[-n+point[0]][point[1]]>data_half):
            n_point_up_half[2]+=1
        else:
            y2=data[-n+point[0]][point[1]]
            y1=data[-n+1+point[0]][point[1]]
            n_point_up_half[2]+=(data_half-y1)/(y2-y1)
            break
    n=0
    while(True):
        #上方向
        n+=1
        if(n+point[0]>len(data[0])-1):
            n_point_up_half[3]=0
            break
        if(data[n+point[0]][point[1]]>data_half):
            n_point_up_half[3]+=1
        else:
            y2=data[n+point[0]][point[1]]
            y1=data[n-1+point[0]][point[1]]
            n_point_up_half[3]+=(data_half-y1)/(y2-y1)
            break
    #乘以格子的长度和宽度
    step_y=site[1][1][0]-site[1][0][0]
    step_x=site[0][0][1]-site[0][0][0]
    n_point_up_half[0]*=step_x
    n_point_up_half[1]*=step_x
    n_point_up_half[2]*=step_y
    n_point_up_half[3]*=step_y
    #求出所有方向上距离的均方均值
    n_point_up_half_end_2=[n**2 for n in n_point_up_half if n!=0]
    return np.sqrt(np.mean(n_point_up_half_end_2))*2*1e9 
'''一、导入数据以及预处理'''
data=np.loadtxt(path,unpack=True)
#获取数据长宽，并重新排列数据成网格形式
#原始数据有三列，分别为x轴坐标，y轴坐标，磁矩值
len_x=np.sum(data[1]==data[1][0])#获取相同的y值有多少，即确定x有多少不同的取值
len_y=np.sum(data[0]==data[0][0])#获取相同的x值有多少，即确定y有多少不同的取值，也即y轴长度

#重新排列后的数据，第一列为横坐标，第二列为纵坐标，第三列为磁矩大小
#其中第三列的磁矩大小是一个二维数据，data_plot[2][y][x]给出了x,y坐标点的磁矩值
data_plot=data.reshape(3,len_y,len_x)
Mz=data_plot[2]
#plt.plot(data_plot[1][:,77],Mz[:,77],'ko-')
#plt.show()
'''二、数据处理'''
Mz_max=np.max(data[2])#存储磁性信号的最大值，用于后续判断
Mz_average=np.median(data[2])
print('磁矩均值为：',Mz_average)
site_skyrmion=get_local_extremum_D2(Mz)
#去除磁矩强度过低的点
site_skyrmion=[point for point in site_skyrmion if Mz[point[0]][point[1]]>(Mz_max*0.1+Mz_average*0.9)]
print('找到了{}个skyrmion'.format(len(site_skyrmion)))

#计算半高宽
half_width=[]
for p in site_skyrmion:
    width=get_half_width(Mz,p,data_plot[:2])
    half_width.append(width)
'''三、绘图'''
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 18.5)
#填色
data_range=[data[0][0],data[0][-1],data[1][0],data[1][-1]]#找出数据范围，以供画图时使用
im1=ax.imshow(data_plot[2],extent=data_range,cmap=plt.cm.hot,origin="lower",interpolation='bilinear',aspect='auto')#extent=[-1,1,-1,1] , interpolation='nearest', 
plt.colorbar(im1)
for i in range(len(site_skyrmion)):
    #画出找到的skyrmion
    site_point=data_plot[:2,site_skyrmion[i][0],site_skyrmion[i][1]]
    ax.text(*site_point,'{:.0f}'.format(half_width[i]))
    ax.plot(*site_point,'b',marker='o', markersize=3)
    #site_point_width=[site_point[0]+(data_plot[0][0][-1]-data_plot[0][0][0])*0.01,site_point[1]]
    print('第%i个,坐标为'%(i+1),site_point,'处的半高宽为%.3fnm,磁矩极值为%.3f'%(half_width[i],Mz[site_skyrmion[i][0]][site_skyrmion[i][1]]))
ax.invert_yaxis()
ax.xaxis.set_ticks_position('top')
print('总体最大值为%.2f,均值为%.2f'%(Mz_max,Mz_average))
#cs=ax.contour(*data_plot,levels=5,colors='black')
#plt.clabel(cs, inline=True, fontsize=8,fmt='%1.4f')



'''四、保存数据'''
#保存图片
plt.savefig(path[:-3]+'png',dpi=300)
plt.show()