# data_processing
数据处理的代码
## 识别斯格明子中心并标注半径
### 实现方法
找到磁矩极大值的点（在x、y方向都是极大值，而且强度值必须达到平均值的一定倍数，可调），在x，y方向向外探索，当强度值小于设定值后就认为到达边界，两者的算术平均值就是标准的半径
![SC_399-sc_phase-fwd_xyz](https://user-images.githubusercontent.com/56717657/188652873-4571939e-3038-4593-bae8-a7dd78dcb533.png)
