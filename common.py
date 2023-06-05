import matplotlib.pyplot as plt
import matplotlib,re
import numpy as np
import os,collections,re,time
from numpy.core.fromnumeric import argmax
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import t
#功能
#1.给定关键词与文件夹，找出所有符合要求的文件(不包括子文件夹)
#2.给定关键词与文件夹，找出所有符合要求的文件(包含子文件夹)
#3.给定函数与范围，画出图像
#4.给定文件夹，画出其所有函数图像
#5
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def find_file(keywords,fold,n_choosed=None,file_type='txt',if_show=1):
    '''给定关键词与文件夹，返回所有符合要求的文件'''
    files_list=os.listdir(fold)#找到要求目录下的所有文件
    if type(keywords)==str:
        #关键词如果是字符串，改成列表
        keywords=[keywords]

    files_list=[f for f in files_list if f[-3:]==file_type]#只找特定类型的文件
    for key in keywords:
        #找到包含关键词的文件
        files_list=[f for f in files_list if key in f]
    if if_show:
        print('找到了',files_list)
    if n_choosed is None:
        return files_list
    else:
        return files_list[int(n_choosed)]
def find_file_walk(keywords,fold,n_choosed=None,file_type='txt',return_fold=False):
    '''给定关键词与文件夹，遍历文件夹及其子文件夹，返回所有符合要求的文件''' 
    # 创建一个栈,一个用来存放原目录路径
    sourceStack = collections.deque()
    sourceStack.append(fold)
    if(return_fold):
        files=[['./',f] for f in find_file(keywords=keywords,fold=fold,if_show=0,file_type=file_type)]#用来存放寻找到的文件
    else:
        files=find_file(keywords=keywords,fold=fold,if_show=0,file_type=file_type)#用来存放寻找到的文件
    while True:
        if len(sourceStack) == 0:
            #清空栈后退出
            break 

        # 将路径从栈的上部取出
        sourcePath = sourceStack.pop()  #sourcePath = sourceStack.popleft()
        # 遍历出该目录下的所有文件和目录
        listName = os.listdir(sourcePath)

        # 遍历目录下所有文件组成的列表,判断是文件,还是目录
        for name in listName:
            sourceAbs = os.path.join(sourcePath, name) # 拼接新的路径
            if os.path.isdir(sourceAbs):
                # 判断是否是目录
                name=sourceAbs.replace(fold,'').lstrip('/').lstrip('\\')
                if(return_fold):
                    #如果返回目录，将文件名与其目录一起返回
                    files.extend([[name,f] for f in find_file(keywords,sourceAbs,if_show=0,file_type=file_type)])#子目录中找到的文件继续添加到文件列表中
                else:
                    files.extend([name+'\\'+f for f in find_file(keywords,sourceAbs,if_show=0,file_type=file_type)])#子目录中找到的文件继续添加到文件列表中
                sourceStack.append(sourceAbs)#目录继续添加到堆栈之中
    print('通过关键词',keywords,'遍历找到了',files)
    if(len(files)>1):
        print('警告，一个关键词找到了多个文件！！！')
    if n_choosed is None:
        return files
    else:
        return files[int(n_choosed)]
def plot_f(f,x_begin=-1,x_end=1,n=1000,if_show=1):
    x_list=np.linspace(x_begin,x_end,n)#生成一系列的X点，用于画图
    y_list=[f(x) for x in x_list]
    plt.plot(x_list,y_list)
    if(if_show):
        plt.show()
        plt.close()
def load_data(name):
    '''给定文件路径，利用正则表达式从文件中找到需要的数据，并返回'''
    with open(name,'r') as f:
        #导入文件
        raw_data=f.read()
    rule=r'@Data[\s\S]+@@END'#文件中数据区域规则，前后有@Data和@@END包围
    data_temp=re.findall(rule,raw_data)
    data=re.findall(r'-?\d+\.\d+E[+-]\d+',data_temp[0])#将每个数据分隔开，数据格式：最开始有负号或没有，一个小数点，至少一个数字数字
    return list(map(eval,data))
def sel_data(data,l=7,n=None):
    '''从找到的数据中返回需要的两列数据'''
    if n is None:
        n=[2,6]#第三列以及第7列
    assert len(data)%l==0,'数据列数输入错误，不能整除'   
    data = np.array(data)
    res=np.reshape(data,(-1,l)) #挑选出自己需要的数据
    res=np.transpose(res[:,n]) #矩阵转置
    return res
def get_data(name):
    raw_data=load_data(name)
    data=sel_data(raw_data)
    return data
def plot_one_fig(data,outname='',title='',xlabel='',ylabel='',ifclose=True,if_print=False):
    #绘制一幅图像
    if(if_print):
        print('处理{}中'.format(outname))
    plt.plot(*data,'ko-')

    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))#设置纵轴刻度为科学计数法
    ax.minorticks_on()#显示次刻度线
    ax.grid(b=True, which='major', lw='1', linestyle='-')
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.5)

    if(xlabel):
        plt.xlabel(xlabel)
    if(ylabel):
        plt.ylabel(ylabel)
    if(title):
        plt.title(title)

    if(outname):
        plt.tight_layout()  #输出图像时能不被挡住
        plt.savefig(outname) #储存图像
    if(ifclose):
        plt.close()    
def plot_mul_fig(data,fig_name='',legend='',title='',xlimt=None,ylimt=None,xlabel=r'$H\ (Oe)$',ylabel=r'$M\times V\ (emu)$',line_color=None,line_style=None,marker='o',n_line_style=2,smooth=0,line_by_number=True,show_grid=True):
    #在一幅图上画很多条曲线
    if type(line_color)==type(None):
        line_color=['k','y','r','b','g','c','m','pink','linen','cyan','silver']
    if type(line_style)==type(None):
        line_style=['-','--',':','-.']
    marker_list=['o','o','.','^']
    markerfacecolor_list=[None,'white',None,'none']
    i_color,i_line_style,i_marker,i_markerfacecolor=0,0,0,0#线颜色与类型初始化
    for i in range(len(data)):
        
        if(smooth):
            xnew = np.linspace(data[i][0].min(), data[i][0].max(), 200)
            spl = interp1d(data[i][0], data[i][1], kind='slinear') 
            x=xnew
            y = spl(xnew)
        else:
            x,y=data[i]
        if(line_by_number):
            i_color=i//n_line_style
            i_line_style=i%n_line_style
        else:
            if(i!=0 and (re.sub(r'[$\\pm+-]','',legend[i])!=re.sub(r'[$\\pm+-]','',legend[i-1]) and (int(re.sub(r'[$\\pm+-]','',legend[i]))<=50 or int(re.sub(r'[$\\pm+-]','',legend[i]))>=54))):
                i_line_style=0
                i_color+=1
                i_marker=0
                i_markerfacecolor=0
            elif i==0:
                i_color=0
            else:
                i_line_style+=1
                i_marker+=1
                i_markerfacecolor+=1
            if i_markerfacecolor>3:
                i_markerfacecolor=0

            #print(int(re.sub(r'[$\\pm+-]','',legend[i])))
            #print(re.sub(r'[$\\pm]','',legend[i]),i_color,i_line_style)
        plt.plot(x,y,color=line_color[i_color],linestyle=line_style[i_line_style],marker=marker_list[i_marker],markerfacecolor=markerfacecolor_list[i_markerfacecolor])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))#设置纵轴刻度为科学计数法
    
    if(show_grid):
        ax.minorticks_on()#显示次刻度线
        ax.grid(b=True, which='major', lw='1', linestyle='-')
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
    if(legend):
        plt.legend(legend,loc="best")
    if(title):
        plt.title(title)
    if(type(xlimt)!=type(None)):
        plt.xlim(xlimt)
    if(type(ylimt)!=type(None)):
        plt.ylim(ylimt)
    if(fig_name):
        plt.tight_layout()#让图像所有元素显示完全
        plt.savefig(fig_name.replace(r'.png','')+'.png')

    plt.show()
    plt.close()
def plt_all(fold,out_fold='fig',xlabel='',ylabel='',fig_type='png',if_title=False,if_print=False):
    #给定文件夹，画出文件夹下每个文件的图
    keyword=['']
    files=find_file_walk(keywords=keyword,fold=fold,return_fold=True)
    '''
    for f in files:
        #获取文件名以及所在目录
        f=fold+'\\'+f
        if('/' in f):
            #得到数据文件所在目录
            out_fold_temp=re.search(r'([.]*/)[\s\S]*\.txt$',f).group(1)
        elif('\\' in f):
            out_fold_temp=re.search(r'([.]*\\)[\s\S]*\.txt$',f).group(1)
        else:
            out_fold_temp=''
        fname=f.replace(out_fold_temp,'')
        out_fold_temp+=out_fold+'\\'
        if not os.path.exists(out_fold_temp):
            #不存在输出文件夹时创建
            os.makedirs(out_fold_temp)
        #获取数据
        data=np.loadtxt(f,unpack=True)
        #绘制图像并保存
        if(if_title):
            title=fname[:-4]
        else:
            title=''
        if(len(data)>0):
            plot_one_fig(data,outname=out_fold_temp+fname[:-3]+fig_type,title=title,xlabel=xlabel,ylabel=ylabel,if_print=if_print)
        else:
            print('{}为空'.format(f))
    '''
    for f in files:
        #获取文件名以及所在目录
        f[0]=fold+'/'+f[0]
        t=time.time()
        if not os.path.exists(out_fold+'/'+f[0]):
            #不存在输出文件夹时创建
            os.makedirs(out_fold+'/'+f[0])
        #获取数据
        data=np.loadtxt(f[0]+'/'+f[1],unpack=True)
        #绘制图像并保存
        if(if_title):
            title=f[1][:-4]
        else:
            title=''
        if(len(data)>0):
            data=np.array(data)
            plot_one_fig(data,outname=f[0]+'/'+out_fold+'/'+f[1][:-3]+fig_type,title=title,xlabel=xlabel,ylabel=ylabel,if_print=if_print)
            print('总时间',time.time()-t)
        else:
            print('{}为空'.format(f))
def get_switch_point(data,point_type=0,n_average=2,n_ignore=1,if_print=1,if_plot=1,s='左边界',c_point='r',c_line='k',label=None,if_ignore=True,unit=1):
    #获取发生翻转的点
    #data:输入数据,如[[1,2,3,4],[1,2,3,4]]
    #point_type:寻找方式，0为找中点，1为找斜率变化最大的点
    #n_average:中点模式中，最大的几个点，n_ignore:最大以及最小的几个值不计算在内
    y=data[1]
    if(if_ignore):
        y_max=np.median(y[np.argsort(data[1])[-n_average-n_ignore:-n_ignore]])
        y_min=np.median(y[np.argsort(data[1])[n_ignore:n_average+n_ignore]])  
    else:
        y_max=np.max(data[1])
        y_min=np.min(data[1])       
    y_center=(y_max+y_min)/2#得到中值
    if point_type==0:
        #认为中间的点是临界点
        f=interp1d(data[0],data[1])#,kind='nearest'
        x_temp=np.linspace(data[0][0],data[0][-1],max(100,int(10*abs(data[0][-1]-data[0][0]))))#让点密集些
        f_inverse=interp1d(f(x_temp),x_temp)#翻转曲线的逆函数，便于寻找对应阻值对应的电流 ,kind='cubic' ,fill_value='extrapolate'  
        x_switch=f_inverse(y_center)#得到中值对应的x值（如磁场，脉冲大小）
        y_switch=y_center
        #plt.close()
        #plt.plot(x_temp,f(x_temp),'ko-')
        #plt.show()
    elif point_type==1:
        #认为斜率最大的点是临界点，取斜率最大的三个点加权平均
        dy_dx=np.array([abs(data[1][i+1]-data[1][i])/abs(data[0][i+1]-data[0][i]) for i in range(len(data[0])-1)])
        x0=np.array([(data[0][i+1]+data[0][i])/2.0 for i in range(len(data[0])-1)])
        temp=np.argsort(dy_dx)[-1]
        dy_dx_max_index=[temp-2,temp-1,temp,temp+1,temp+2]
        x_switch=np.sum(x0[dy_dx_max_index]*dy_dx[dy_dx_max_index])/np.sum(dy_dx[dy_dx_max_index])
        y_switch=interp1d(data[0],data[1])(x_switch)
    if(if_print):
        print('纵向最小值为{:.3g}Ω,最大值为{:.3g}Ω,高度为{:.3g}Ω'.format(y_min,y_max,y_max-y_min))
        print('横轴翻转{}：{:.2f}，对应电阻为：{:.3g}Ω\n'.format(s,x_switch*unit,y_switch))
    if if_plot:
        plt.plot(data[0],data[1],'{}o-'.format(c_line))
        plt.plot(x_switch,y_switch,'{}o'.format(c_point))
    return x_switch,(y_max-y_min)/2

def get_height_old(data,mode=1):
    #得到翻转的高度
    if(data[0][0]*2<max(data[0])+-min(data[0])):
        #此时初始值更接近最小值，分割点为最大值
        x_index=np.argsort(data[0])#越小的数的索引在越前面
        split_point_index=x_index[-1]#分割点的索引
    data1=data[:,:split_point_index]
    data2=data[:,split_point_index:] 

    if(mode==0):
        #取平均值
        x_choose=np.where(data1[0]<0)#选出负脉冲对应的状态
        data_average=data1[:,x_choose]
        y_min=np.median(data_average[1])

        x_choose=np.where(data2[0]>0)#选出正脉冲对应的状态
        data_average=data2[:,x_choose]
        y_max=np.average(data_average[1])
    if(mode==1):
        #最大的5个数的中位数与最小的5个数的中位数
        n_median=5
        y_index_list=np.argsort(data1[1])
        y_max=np.median(data1[1][y_index_list[:n_median]])

        y_min=np.median(data1[1][y_index_list[-n_median-1:]])
    return y_max-y_min
def get_shift(data,unit=['mA',1000],c_point_list=['r','g'],c_line_list=['k','b'],if_print=1,legend=None,if_show=1,*args,**kwargs):
    #得到中心点的偏移
    if(data[0][0]*2<max(data[0])+-min(data[0])):
        #此时初始值更接近最小值，分割点为最大值
        x_index=np.argsort(data[0])#越小的数的索引在越前面
        split_point_index=x_index[-1]#分割点的索引
    data1=data[:,:split_point_index]
    data2=data[:,split_point_index:]
    x_switch1,y_high1=get_switch_point(data1,s='右边界',c_point=c_point_list[0],c_line=c_line_list[0],if_print=if_print,unit=unit[1],**kwargs)
    x_switch2,y_high2=get_switch_point(data2,s='左边界',c_point=c_point_list[1],c_line=c_line_list[1],if_print=if_print,unit=unit[1],**kwargs)

    x_height=abs(x_switch1-x_switch2)/2
    y_high=(y_high1+y_high2)/2.0
    x_center=(x_switch1+x_switch2)/2.0
    if(if_print):
        print('*'*5,'小结','*'*5)
        print('中心偏移为{:.2f}{}'.format(x_center*unit[1],unit[0]))#*1000
        print('横向翻转均值为{:.2f}{}'.format(x_height*unit[1],unit[0]))#*1000
        print('霍尔电阻为{:.3g}Ω'.format(y_high))
    if(if_show):
        if(type(legend)!=type(None)):
            lines=plt.gca().get_lines()
            plt.legend([lines[0],lines[4]],legend)
        plt.show()
        plt.close()
    return np.round(np.array([x_center*unit[1],x_height*unit[1],y_high]),2)
def get_loop_shift(data,if_print=1,if_show=1,unit=['Oe',1],*args,**kwargs):
    #给定两个电流值的AHE数据，返回正电流有效场的大小，两个电流的矫顽力以及霍尔电阻
    data_pos=data[0]
    data_neg=data[1]
    x_center_pos,x_height_pos,y_high_pos=get_shift(data_pos,if_print=0,if_show=0,c_line_list=['k','k'],c_point_list=['b','b'],unit=unit,**kwargs)
    x_center_neg,x_height_neg,y_high_neg=get_shift(data_neg,if_print=0,if_show=if_show,c_line_list=['g','g'],c_point_list=['y','y'],unit=unit,**kwargs)
    Hz_eff=(x_center_neg-x_center_pos)/2
    if if_print:
        print('*'*5,'loop shift 总结:','*'*5)
        print('SOT在z方向有效场为:{:.2f}Oe'.format(Hz_eff))
        print('正负电流下的矫顽力分别为{:.2f}Oe,{:.2f}Oe'.format(x_height_pos,x_height_neg))
        print('正负电流霍尔电阻变化值为{:.2f}欧姆,{:.2f}欧姆'.format(y_high_pos,y_high_neg))
        print('*'*20,'END','*'*20)
    
    return [round(Hz_eff,2),[x_height_pos,x_height_neg],[y_high_pos,y_high_neg]]

def remove_skew(data,error=0.2,return_number=0):
    #去除数据跳点，用两边平均值替代
    skew_count=0#跳点计数
    y=np.copy(data[1])
    x=np.copy(data[0])
    l=len(y)
    #处理中间的数据点
    for i in range(1,l-1):
        if x[i]<max(x):
            if( abs(y[i]/y[i-1]-1)>error and abs(y[i]/y[i+1]-1)>error and (y[i]-y[i-1])*(y[i]-y[i+1])>0):
                #只有与两边的差别都达到要求，且误差的符号相同时，才认为是跳点
                print('在x={},y={}处有跳点,y新值为{}'.format(x[i],y[i],(y[i-1]+y[i+1])/2))
                #print('data_new',(y[i-1]+y[i+1])/2)
                #有跳点，取平均值
                skew_count+=1
                y[i]=(y[i-1]+y[i+1])/2

    #处理两头的数据点
    if (abs(y[0]/y[1]-1)>error and abs(y[0]/y[2]-1)>error and (y[0]-y[1])*(y[0]-y[2])>0):
        print('在x={},y={}处有跳点,y新值为{}'.format(x[0],y[0],y[1]-(y[2]-y[1])/(x[2]-x[1])*(x[1]-x[0])))
        skew_count+=1
        y[0]=y[1]-(y[2]-y[1])/(x[2]-x[1])*(x[1]-x[0])
    if (abs(y[-1]/y[-2]-1)>error and abs(y[-1]/y[-3]-1)>error and (y[-1]-y[-2])*(y[-1]-y[-3])>0):
        print('在x={},y={}处有跳点,y新值为{}'.format(x[-1],y[-1],y[-2]+(y[-2]-y[-3])/(x[-2]-x[-3])*(x[-1]-x[-2])))
        skew_count+=1
        y[-1]=y[-2]+(y[-2]-y[-3])/(x[-2]-x[-3])*(x[-1]-x[-2])
    if(return_number):
        return np.array([x,y]),skew_count
    else:
        return np.array([x,y])     
def remove_line_background(data,x_start=100,mode=0,point=10,if_center=0,if_norm=0,if_remove_repiao=0):
    #去除数据的线性背底
    l=len(data[0])
    if(data[0][0]>data[0][l//2]):
        #print('检测到磁场不是从负场扫起，将数据左右颠倒')
        data=data[:,::-1]
    if(if_remove_repiao):
        delta=data[1][-1]-data[1][0]
        for i in range(l):
            data[1][i]-=delta*i/l
    
    data1=data[:,:l//2]
    data2=data[:,l//2:]
    
    #找到要求范围的数据
    if(mode==0):
        #给定磁场范围
        f1=interpolate.interp1d(*data1,kind='cubic',fill_value="extrapolate")
        f2=interpolate.interp1d(*data2,kind='cubic',fill_value="extrapolate")
        #得到线性范围内的数据点
        #右侧
        x1_new_right=np.linspace(x_start,max(data1[0]))
        x2_new_right=np.linspace(x_start,max(data2[0]))

        y1_new_right=f1(x1_new_right)
        y2_new_right=f2(x2_new_right)

        #左侧
        x1_new_left=np.linspace(min(data1[0]),-abs(x_start))
        x2_new_left=np.linspace(min(data2[0]),-abs(x_start))
        y1_new_left=f1(x1_new_left)
        y2_new_left=f2(x2_new_left)      
    if(mode==1):
        #给定点数
        x1_new_left,y1_new_left=data1[:point]
        x2_new_left,y2_new_left=data2[-point:]
        x1_new_right,y1_new_right=data1[-point:]
        x1_new_right,y2_new_right=data2[:point]
    #拟合得到直线斜率
    k1_left,b1_left=np.polyfit(x1_new_left, y1_new_left, 1) 
    k2_left,b2_left=np.polyfit(x2_new_left, y2_new_left, 1) 
    k1_right,b1_right=np.polyfit(x1_new_right, y1_new_right, 1) 
    k2_right,b2_right=np.polyfit(x2_new_right, y2_new_right, 1) 

    k_right,k_left=(k1_left+k2_left)/2,(k1_right+k2_right)/2
    b_right,b_left=(b1_left+b2_left)/2,(b1_right+b2_right)/2
    k=(k_right+k_left)/2
    x_new=data[0]
    y_new=data[1]-k*data[0]
    if(if_center):
        y_new-=(b_right+b_left)/2
    if(if_norm):
        y_new/=abs(b_right-b_left)/2
    return np.array([x_new,y_new])
def remove_odd_function(data):
    #移除偶函数
    max_index=np.argsort(data[0])[0]
    #把霍尔电阻移到中间
    data[1]=data[1]-(max(data[1])+min(data[1]))*0.5
    #去除热漂
    l=len(data[0])
    delta_y=interp1d(*data[:,max_index:],fill_value="extrapolate")(data[0][0])-data[1][0]
    for i in range(l):
        data[1][i]-=delta_y*i/(l-1)

    #以磁场最大值为界，隔成两段
    data1=data[:,:max_index]
    data2=data[:,max_index:]

    #插值
    f1=interp1d(*data1,fill_value="extrapolate")
    f2=interp1d(*data2,fill_value="extrapolate")
    #扣除偶函数
    y_new1=(data1[1]-f2(-data1[0]))/2#(f(x)-f'(-x))/2
    y_new2=(data2[1]-f1(-data2[0]))/2
    y_all=np.append(y_new1,y_new2,axis=0)
    data_all=np.array([data[0],y_all])
    return data_all
def remove_data_by_point(data_back,data):
    #从一个数据点中扣去另一个数据点
    x=np.copy(data[0])
    f_data_back=interpolate.interp1d(*data_back,kind='linear',fill_value="extrapolate")
    y=data[1]-f_data_back(x)
    return np.array([x,y])
def remove_init(data):
    pass
def remove_repeat_x(data):
    index_del=[]

    if(type(data[0]) in [list,np.ndarray]):
        x=data[0]
        dim=1
    else:
        x=data
        dim=0
    for i in range(len(x)-1):
        if(x[i+1]==x[i]):
            index_del.append(i+1)
    data=np.delete(data,index_del,dim)
    return data
def remove_background_by_point(data_background,data,num=None,x0=None):
    #去除由测量给定的背底
    index_max_back=np.argmax(data_background[0])#找到磁场最大值索引
    index_max_data=np.argmax(data[0])
    data1=data[:,:index_max_data]
    data2=data[:,index_max_data:]
    if(num is not None):
        f1=interpolate.interp1d(*data1,kind='linear',fill_value="extrapolate")
        f2=interpolate.interp1d(*data2,kind='linear',fill_value="extrapolate")
        x1=np.linspace(data1[0][0],data1[0][-1],num)
        x2=np.linspace(data2[0][0],data2[0][-1],num)
        data1=np.array([x1,f1(x1)])
        data2=np.array([x2,f2(x2)])
    if(x0 is not None):
        f1=interpolate.interp1d(*data1,kind='linear',fill_value="extrapolate")
        f2=interpolate.interp1d(*data2,kind='linear',fill_value="extrapolate")
        index_max_data=np.argmax(x0)
        x1=x0[:index_max_data]
        x2=x0[index_max_data:]
        data1=np.array([x1,f1(x1)])
        data2=np.array([x2,f2(x2)])
    data_background1=data_background[:,:index_max_back]
    data_background2=data_background[:,index_max_back:]
    data_new1=remove_data_by_point(data_background1,data1)
    data_new2=remove_data_by_point(data_background2,data2)
    return np.append(data_new1,data_new2,axis=1)
def find_min_max(data,precision=2,error=1e-7):
    '''寻找数据中极小值和极大值所在位置'''
    seat_min=[]#记录极小值位置
    seat_max=[]#记录极大值位置
    for i in range(len(data)):
        sum_min=0
        sum_max=0
        if i>precision and i+precision<len(data):#确保数组不越界
            for j in range(precision):#找到极小值
                sum_min+=(data[i]<=data[i-j-1]+error)
                sum_min+=(data[i]<=data[i+j+1]+error)
            if  sum_min>=precision*2:
                if len(seat_min)>0 and i-seat_min[-1][0]<=1:#连续的两个极小值，取后面那个
                    seat_min[-1][0]=i
                    #print('double min')#测试用
                else:
                    seat_min.append([i,data[i]])

            for j in range(precision):#找到极大值
                sum_max+=(data[i]>=data[i-j-1]-error)
                sum_max+=(data[i]>=data[i+j+1]-error)
            if  sum_max>=precision*2:
                if len(seat_max)>0 and i-seat_max[-1][0]<=1:#连续的两个极大值，取后面那个
                    seat_max[-1][0]=i
                    #print('double max')#测试用
                else:
                    seat_max.append([i,data[i]])
    l=len(data)

    if(0.9<= data[-1]/max(data)<=1.1):
        seat_max.append([l-1,data[l-1]])
    if(0.9<= data[-1]/min(data)<=1.1):
        print('最后一个值是最小值')
        seat_min.append([l-1,data[l-1]])  

    if(0.9<= data[0]/max(data)<=1.1):
        seat_max=[[0,data[0]]]+seat_max
    if(0.9<= data[0]/min(data)<=1.1):
        seat_min=[[0,data[0]]]+seat_min   
     
    seat_min,seat_max=np.array(seat_min),np.array(seat_max)
    #print seat_min,seat_max#测试用，输出找到的极小值和极大值数据
    return list(map(int,seat_min[:,0])),list(map(int,seat_max[:,0]))
def select_data(data,n=0,select_half=False,data_half=False):
    '''输出数据中第n个极小值（不包括第一个数据点）到随后的第一个极大值，以及此极大值到随后的极小值的数据,假设数据是从最小值开始测 '''   
    seat_min,seat_max=find_min_max(data[0])
    sign=0

    seat=sorted(np.append(seat_min,seat_max))   
    num=int(2*n+sign)
    #print num
    #print(seat[num],seat[num+1])
    data_new=data[:,seat[num]:seat[num+1]]#第n个极小值（不包括第一个数据点）到随后的第一个极大值的数据
    
    if(select_half):
        x1,y1=data_new
        data_new=np.array([np.append(x1,x1[::-1]),np.append(y1,-y1+y1[0]+y1[-1]),])
    elif(data_half):
        return data_new
    else:
        x1,y1=data_new
        x2,y2=data[0,seat[num+1]:seat[num+2]],data[1,seat[num+1]:seat[num+2]]#第n+1个极大值（不包括第一个数据点）到随后的第一个极小值的数据
        data_new=np.array([np.append(x1,x2),np.append(y1,y2)])
    return data_new
def move_center_to_zero(data,n=5,if_guiyi=0,center=0,tongxiang=1):
    '''将数据点的纵轴移动到零点处'''
    #得到平均最大值与最小值
    if(tongxiang):
        l=len(data[1])
        if(data[1][l//4]<data[1][l*3//4]):
            data[1]*=-1
    x,y=np.copy(data)
    y_max=np.mean(y[np.argsort(data[1])[-n:]])
    y_min=np.mean(y[np.argsort(data[1])[:n]])
    if(center==0):
        y_center=(y_max+y_min)*0.5
    elif(center==1):
        y_center=(y[0]+y[-1])/2
    y=y-y_center
    if(if_guiyi):
        y=y*2/(y_max-y_min)
    print('最大值与最小值之差为：%.2e'%(y_max-y_min),'中心点偏移了%.2e'%y_center)
    return np.array([x,y])
def nor_data(data):
    #将数据处理为标准数据,并返回两段
    if(len(data)>4):
        data=np.transpose(data)
    data=move_center_to_zero(data,tongxiang=0)
    l=len(data[0])

    index_max_data=np.argmax(data[0])
    index_min_data=np.argmin(data[0])

    index_data=index_max_data
    if(abs(index_min_data/l-0.5)<0.2):
        #数据倒过来了
        index_data=index_min_data
    print(index_max_data,l)
    if(index_data>l*3//4):
        data1=np.array(data[:,:index_data])
        data2=np.array([-data1[0],-data1[1]])
        print('只有前半段数据')

    elif(index_data<len(data[0])*1//4):
        data2=np.array(data[:,index_data:])
        data1=np.array([-data2[0],-data2[1]])
        print('只有后半段数据')
    else:
        data1=np.array(data[:,:index_data])
        data2=np.array(data[:,index_data:])
        if(data[0][0]>data[0][index_data]):
            data1,data2=data2,data1
        print('完整数据')
    return data1,data2
def get_height(data,n=7,height_type=0,n_ignore=1,if_plot=True,n_point=3,return_data=False):
    x,y=np.copy(data)

    if(height_type==0):
        #最大值减最小值
        y_max=np.mean(y[np.argsort(data[1])[-n:-n_ignore]])
        y_min=np.mean(y[np.argsort(data[1])[n_ignore:n]])
        height=(y_max-y_min)*0.5
    elif(height_type==1):
        data1_raw,data2_raw=nor_data(data)
        data1=remove_data_LF(data1_raw,H_limt=(data1_raw[0][0],data1_raw[0][n_point]))
        data2=remove_data_LF(data1_raw,H_limt=(data1_raw[0][-n_point-1],data1_raw[0][-1]))

        (k1,b1),pcov1=curve_fit(lambda x,k,b: k*x+b,*data1)
        (k2,b2),pcov2=curve_fit(lambda x,k,b: k*x+b,*data2)
        tp=3#99%不确定度
        uat1=np.sqrt(pcov1[1][1]/len(data1[0]))*tp
        uat2=np.sqrt(pcov2[1][1]/len(data2[0]))*tp
        uat1_k=np.sqrt(pcov1[0][0]/len(data1[0]))*tp
        error=(uat1+uat2)/2
        #print(k1,b1,uat1_k,uat1)
        #height_min=interpolate.interp1d(*data1,kind='linear',fill_value="extrapolate")(0)
        #height_max=interpolate.interp1d(*data2,kind='linear',fill_value="extrapolate")(0)
        height=(abs(b2)+abs(b1))/2#(height_max-height_min)/2
        if(if_plot):
            plt.plot(*data1_raw,'ko')
            plt.plot(*data2_raw,'ro')
            plt.plot(data1_raw[0],k1*data1_raw[0]+b1,'k-')
            plt.plot(data2_raw[0],k2*data2_raw[0]+b2,'r-')
            plt.show()
        print('height',height,'error',error)
        if return_data:
            data1_raw[1]=data1_raw[1]-data1_raw[0]*k1
            data2_raw[1]=data2_raw[1]-data2_raw[0]*k2
            return height,np.append(data1_raw,data2_raw,axis=1)
    return height
def get_height_zero(data,n=7,height_type=0,n_ignore=1):
    #获取剩磁大小
    x,y=np.copy(data)

    index_max_data=np.argmax(x)#找到磁场最大值索引
    data1=data[:,:index_max_data]
    data2=data[:,index_max_data:]

    height_min=interpolate.interp1d(*data1,kind='linear',fill_value="extrapolate")(0)
    height_max=interpolate.interp1d(*data2,kind='linear',fill_value="extrapolate")(0)
    height_zero=(height_max-height_min)/2#(height_max-height_min)/2
    return height_zero
def remove_data_LF(data,H_limt=None,H_max=None):
    #取小场下数据
    if(len(data)==2):
        data=np.transpose(data)
    
    data_new=np.array([dat for dat in data if dat[0]>=min(H_limt) and dat[0]<=max(H_limt)])
    data_new=np.transpose(data_new)
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
    if max(abs(delta_list))>np.median(abs(delta_list))*10:
        if(data[0][0]<data[0][-1]):
            data_new=data[:,:index_change_fast-1]
        else:
            data_new=data[:,:index_change_fast-1]
    else:
        data_new=data[:,:]
    if(if_print):
        print('raw',data[:,:5])
        print('res:',data_new[:,:5])
    return data_new
def sep_data(data):
    #分离数据    
    x=data[0]
    index_max_data=np.argmax(x)#找到磁场最大值索引
    data1=data[:,:index_max_data]
    data2=data[:,index_max_data:]
    return data1,data2
def Ms_t_dead(dat,s=''):
    print(s)
    k,delta_k,=dat[0]
    b,delta_b=dat[1]
    print('磁化强度',k/1e-8/0.71/0.71,'误差',delta_k/1e-8/0.71/0.71)
    print('死层厚度',-b/k,'误差',-delta_b/k)
def cal_xi(Mst,beta):
    #计算自旋流效率
    import scipy.constants as sc
    xi_DL=2*sc.e/sc.hbar*Mst*beta
    print(xi_DL)
def cal_xi_lp(Mst,beta):
    #计算自旋流效率
    import scipy.constants as sc
    xi_DL=2*sc.e/sc.hbar*2/np.pi*Mst*beta
    print(xi_DL)
def remove_data_Grubbs(data,p=0.95):
    #使用格鲁布斯方法去除异常值
    alpha=1-p
    N=len(data)
    t_p=abs(t.ppf(alpha/2/N,N-2)) 
    G=(N-1)/np.sqrt(N)*np.sqrt(t_p**2/(N-2+t_p**2))

    x_ave=np.average(data)
    sigma=np.std(data)*np.sqrt(N/(N-1))

    index_list=[]
    x_new=[]
    for i in range(len(data)):
        #数据的剔除
        x_one=data[i]
        if abs(x_one-x_ave)/sigma>G:
            index_list.append(i)
        else:
            x_new.append(x_one)
    return x_new
def cal_uAt(x,p=0.95):
    #求x的平均值与p置信度下的不确定度（使用t分布，针对正态分布的数据）

    x_new=remove_data_Grubbs(x)#去除异常值
    l_old,l_new=len(x),len(x_new)
    t_p=t.interval(p,l_new)[1]#计算t-p
    x_ave=np.average(x_new)
    u_A=np.std(x_new)/np.sqrt(l_old-1)
    u_At=u_A*t_p#拓展不确定度
    
    print('前后数据点个数为',l_old,l_new,'平均值为',x_ave,'%.0f%%概率下误差为'%(p*100),u_At)
    return x_ave,u_At
if __name__=='__main__':
    #Mo40 [[8.624e-6,0.55e-6],[-4.8083e-5,7.409e-6]] 
    #V30Mo10 [[6.244e-6,1.675e-7],[-1.334e-5,2.251e-6]]  
    #		
    #		
    #as_grown
    #Mo40 [[5.50865E-6 , 1.90291E-7],[-1.27744E-5 , 2.55728E-6]]
    #V30Mo10[[5.36177E-6 , 2.41902E-7],[-9.61158E-6 , 3.25087E-6]]
    #V40 [[4.86415E-6 , 4.17501E-8],[-1.14046E-5 , 5.61072E-7]]
    if 0:
        dat_Mo40=[[5.50865E-6 , 1.90291E-7],[-1.27744E-5 , 2.55728E-6]]  
        dat_V30Mo10=[[5.36177E-6 , 2.41902E-7],[-9.61158E-6 , 3.25087E-6]]
        dat_V40=[[4.86415E-6 , 4.17501E-8],[-1.14046E-5 , 5.61072E-7]]
        dat_Ta30Mo10=[[5.50441E-6 , 1.40671E-7],[-1.49406E-5 , 1.89045E-6]]
        dat_Ta40=[[5.10297E-6 , 1.16647E-7],[-1.40212E-5 , 1.5676E-6]]
    dat_Mo40=[[8.62398E-6 , 5.51335E-7],[-4.8083E-5 , 7.40929E-6]]  
    dat_V30Mo10=[[6.24369E-6 , 1.67523E-7],[-1.33351E-5 , 2.25132E-6]]
    dat_V40=[[9.35663E-6 , 1.05338E-6],[-7.73193E-5 , 1.59735E-5]]
    dat_Ta30Mo10=[[9.68977E-6 , 5.68317E-7],[-6.86717E-5 , 7.6375E-6]]
    dat_Ta40=[[9.66785E-6 , 3.03603E-7],[-7.9657E-5 , 4.34334E-6]]
				
    dat=[dat_Mo40,dat_V30Mo10,dat_V40,dat_Ta30Mo10,dat_Ta40]
    for da in dat:
        pass
        #Ms_t_dead(da)
    beta_mA=np.array([[1.667,0.055],[3.561,0.27],[0.214,0.026]])
    xishu=1e-3/20e-6/4e-9/1e10  #MA/cm^2
    ms_t=np.array([4.6e-5,2.92e-5,5.12e-5])/1e3/0.0071/0.0071
    beta_MA_cm2=beta_mA/xishu
    print(beta_MA_cm2)
    beta_MA_cm2_val=beta_MA_cm2[:,0]
    beta_MA_cm2_err=beta_MA_cm2[:,1]
    cal_xi(ms_t,beta_MA_cm2_val/1e10/1e4)
    cal_xi(ms_t,beta_MA_cm2_err/1e10/1e4)
    cal_xi_lp(ms_t[0],np.array([0.2746,0.025])*4e-9*20e-6*1000/10000) #  Mo 0.2746,0.025  0.234,0.012  Ta 0.68229,0.072
    #print(1e-3/20e-4/4.5e-7/1e6*37)
    #time_dp=np.array([10,11,13,15])
    #print((time_dp+1.9)*0.602)
    #print(7.532e-6/1e-8/(0.49+3.14*0.1*0.1),7.6e-6/1e-8/(0.49+3.14*0.1*0.1))
    #print(28.339/7.532,36.345/7.6)    
