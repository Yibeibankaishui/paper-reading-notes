# 文献阅读

[toc]

---

## Nerf-based SLAM

[知乎nerf-slam综述](https://zhuanlan.zhihu.com/p/555996624)

## SLAM

### 0. SLAM和深度学习（推文）

[SLAM 和深度学习（微信推送）](https://mp.weixin.qq.com/s/3YYZJbt3xf6iHqz7kF4Gpg)

#### 端到端视觉里程计

##### SfMLearner

[Unsupervised Learning of Depth and Ego-Motion from Video](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf)

##### SfM-Net

[SfM-Net: Learning of Structure and Motion from Video](https://arxiv.org/pdf/1704.07804.pdf)

##### DeMoN

[DeMoN: Depth and Motion Network for Learning Monocular Stereo](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ummenhofer_DeMoN_Depth_and_CVPR_2017_paper.pdf)

#### 相机重定位 relocalization

##### PoseNet

[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://openaccess.thecvf.com/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)

#### 语义SLAM

##### CNN-SLAM

[CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tateno_CNN-SLAM_Real-Time_Dense_CVPR_2017_paper.pdf)

[UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/pdf/1709.06841.pdf)

##### VSO

[VSO: Visual Semantic Odometry](https://openaccess.thecvf.com/content_ECCV_2018/papers/Konstantinos-Nektarios_Lianos_VSO_Visual_Semantic_ECCV_2018_paper.pdf)

#### 特征点提取与匹配

##### LIFT

[LIFT: Learned Invariant Feature Transform](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_28)

##### SuperPoint

[SuperPoint: Self-Supervised Interest Point Detection and Description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)

##### DeepSLAM

[Toward Geometric Deep SLAM](https://arxiv.org/pdf/1707.07410.pdf)

##### VINet

VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem

### 1. Spline Fusion

[***Spline Fusion: A continuous-time representation for visual-inertial fusion with application to rolling shutter cameras***](http://www.bmva.org/bmvc/2013/Papers/paper0093/paper0093.pdf)

#### 动机

应对 **rolling shutter camera**（[果冻效应（wiki）](https://en.wikipedia.org/wiki/Rolling_shutter)）存在的问题（相机逐行曝光导致图像畸变）和尺度漂移（**Scale drift**）。以及多传感设备的不同步性(**unsynchronize**)，建立一个运动轨迹的**连续时间模型**，仅使用提取到的特征轨迹（feature tracks）就可以对各项参数进行联合优化。

#### 方法

对平移和旋转采用B样条来参数化

用累计基本函数来表达B样条Representing **B-Splines with cumulative basis functions**:

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220725183706642.png?token=AKAJBVLTHJMUIVPJIXCEY6TDJD4AM" alt="image-20220725183706642" style="zoom:50%;" />

这是一个在$\mathbb{SE}(3)$空间上二阶连续（$C^2$-continuous）的模型。固定$k=4$，可以得到导数：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220725201141784.png?token=AKAJBVIIX2QLD7FLFH7M72TDJD4AO" alt="image-20220725201141784" style="zoom:50%;" />

帧a到帧b图像坐标的重投影，其中用逆深度$\rho$来表征特征点空间位置：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220725192355106.png?token=AKAJBVM3OF45U5XZFTA5TIDDJD4AQ" alt="image-20220725192355106" style="zoom:50%;" />

利用B-Spline得到的加速度、角速度值：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220725192527469.png?token=AKAJBVKVFH6DCQU4VJGGBU3DJD4AS" alt="image-20220725192527469" style="zoom:50%;" />

代价函数：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220725191854949.png?token=AKAJBVILZWZB6W2HML6MXCLDJD4AU" alt="image-20220725191854949" style="zoom:43%;" />

最小化代价函数来优化对参数的估计。参数可以包括样条控制点、相机内参、路标逆深度、IMU Bias等。

#### 启发

给定任意轨迹，我们都可以用B-Spline插值的方式得到一条二阶连续的曲线，对其求导可以得到IMU的数据（加速度，角速度）。



References: [贝塞尔曲线](https://zhuanlan.zhihu.com/p/136647181)，[B样条曲线](https://zhuanlan.zhihu.com/p/139759835)

---

### 2. VINS-Mono

***[VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator](https://arxiv.org/pdf/1708.03852.pdf)***

#### Motivation



#### Method



#### Ideas









### 3. Loam

激光SLAM

[LOAM: Lidar Odometry and Mapping in Real-time](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)

[LOAM原理解析（知乎）](https://zhuanlan.zhihu.com/p/111388877)

[loam论文和代码解读（CSDN）](https://blog.csdn.net/robinvista/article/details/104379087)

#### Motivation

雷达在静止状态下，能获得精准的3D建图；在实际运用中，雷达往往处于运动中，会带来点云的mis-registration（误匹配）。传统方法是使用额外的传感器（GPS/INS, encoders）或者视觉里程计。

本文考虑使用2轴6自由度lidar作为低漂移里程计来建图 creating maps with low- drift odometry using a 2-axis lidar moving in 6-DOF

#### Method

* 总体框架

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815131229818.png?token=AKAJBVM672YYUVAWFUPXKD3DJD4AW" alt="image-20220815131229818" style="zoom:50%;" />

将算法分成两部分 Lidar Odometry 和 Lidar Mapping。

Lidar Odometry 作为高频的里程计来估计lidar的速度，实现**粗定位**

Lidar Mapping 在较低频率下进行点云的匹配和配准，实现**精定位**

* 点云线面特征提取 *Feature Point Extraction*

  1. 按线数分割

  2. 计算曲率

     <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815132928134.png?token=AKAJBVORY3NWBZYLN7LJ3G3DJD4A2" alt="image-20220815132928134" style="zoom:35%;" />

  3. 按曲率大小筛选点

     将特征点分为两大类：平面点 planar points 和边缘点 edge points

     基于曲率$c$，有最大$c$的是边缘点，最小$c$的是平面点（同时要超出选择阈值）

  

* 特征匹配 *Finding Feature Point Correspondence*

  将第$k$次扫描的点云$P_k$，分为边缘点集合$\mathcal{E}_k$和平面点集合$\mathcal{H}_k$

  要得到$P_k$到$P_{k+1}$之间的变换关系，也就是$\mathcal{E}_k$与$\mathcal{E}_{k+1}$、$\mathcal{H}_k$与$\mathcal{H}_{k+1}$的对应关系

  为了方便处理，我们将所有的点重投影到每一帧的初始时刻，这样在这一帧中的所有点都可以得到对应的姿态变换信息。

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815134810460.png?token=AKAJBVOHPXPB4K5MC677FPLDJD4A4" alt="image-20220815134810460" style="zoom:50%;" />

  对于边缘点，选取**三个点**，求**点到线**的最近距离

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815133930870.png?token=AKAJBVKXR7GUGYXBERDMIFTDJD4A6" alt="image-20220815133930870" style="zoom:50%;" />

  对于平面点，选取**四个点**，求**点到面**的最近距离

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815133950475.png?token=AKAJBVN46424DB7LNDXNYE3DJD4BA" alt="image-20220815133950475" style="zoom:50%;" />

* 姿态解算 *Motion Estimation*

  构建优化问题，用LM方法求解

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815135318660.png?token=AKAJBVMKLODF3KHRTZHK4ODDJD4BC" alt="image-20220815135318660" style="zoom:50%;" />

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815135329843.png?token=AKAJBVL4KWIHDOQEFIEZPLLDJD4BE" alt="image-20220815135329843" style="zoom:50%;" />

  堆叠起来：

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815135404323.png?token=AKAJBVPWBLR673HBDVL7G2LDJD4BG" alt="image-20220815135404323" style="zoom:50%;" />

  雅可比：$\bold J=\frac{\partial f}{\partial \bold T^L_{k+1}}$

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220815135506788.png?token=AKAJBVOFZ7BQPD7C65X2E2DDJD4BI" alt="image-20220815135506788" style="zoom:50%;" />

#### Ideas

LOAM的点云特征的提取方法十分经典

**优点**：

* 新颖的特征提取方式（边缘点和平面点）
* 运动补偿（时间戳）
* 融合了scan-to-scan（odometry）和map-to-map（mapping）的思想

**缺点**：

* 没有后端优化
* 不能处理大规模的旋转变换（旋转向量的求解）

### 4. A-loam





### 5. F-loam

[F-LOAM : Fast LiDAR Odometry and Mapping](https://arxiv.org/pdf/2107.00822.pdf)



### 6. $\sqrt{\text{BA}}$ for large-scale reconstruction

[Square Root Bundle Adjustment for Large-Scale Reconstruction](https://openaccess.thecvf.com/content/CVPR2021/papers/Demmel_Square_Root_Bundle_Adjustment_for_Large-Scale_Reconstruction_CVPR_2021_paper.pdf)

![image-20220825213029376](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825213029376.png?token=AKAJBVJ5QD2A657TFJG3KVDDJD4BO)

#### Motivation

解决针对SfM/3D重建的***大规模***BA问题

传统做法：用 Schur complement ，计算正规方程 normal equations，将大问题转化为小的子问题，将系统转变为RCS (reduced camera system)，更利于求解

#### Method

* nullspace marginalization

  利用QR分解，将原问题 project 到$J_l$的 nullspace 上，**规避了正规方程（Hessian matrix）的计算**

  ![image-20220825223717624](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825223717624.png?token=AKAJBVK5BNPBDG5SS2DZQC3DJD4BQ)

  先求$\Delta x_p$

  ![image-20220825223707168](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825223707168.png?token=AKAJBVLMH22532TNU4FPCTLDJD4BS)

  然后求$\Delta x_l$

  ![image-20220825223814186](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825223814186.png?token=AKAJBVOQ6EJ6JEF7CSPTPYTDJD4BU)

  相比于Schur做法中用双精度的Ceres solver，本文可以采用单精度float类型，保证**数值稳定性**的同时，加快了**运行速度**

  同时证明，与Schur补方法具有**代数等价性**

* Implementation strategy

  用如下形式的 landmark block 来存储。经过几次 Givens Rotation 来计算QR分解。下半部分是RCS (reduced camera system)，上半部分是 back substitution。

  Givens Rotation 可以对每一块**并行**操作，有加速作用。

  ![image-20220825222056393](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825222056393.png?token=AKAJBVLHUX2LIREVJBLVPKTDJD4BW)

* LM damping

  ![image-20220825222242241](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220825222242241.png?token=AKAJBVJQWWOWBDHRF6ZRQILDJD4BY)

  对于 landmark damping，在先前QR分解完的矩阵下面添加$3\times3$的子块，然后经6次Givens Rotation，再次边缘化。

  当优化出错时，只需要乘以$Q_\lambda^T$（回滚操作），而**不需要重新计算H矩阵**（Schur Trick的做法）

#### 为什么会更快？

作者在demo视频下方给出的答案：

>  We believe what contributes to the better runtime is that the solver can do more of the linear algebra operations on dense memory blocks (the landmark blocks), which is good for SIMD (as Guan points out) but maybe also CPU cache performance. 

[QR分解wiki](https://zh.wikipedia.org/zh-cn/QR%E5%88%86%E8%A7%A3)

[SIMD简述](https://zhuanlan.zhihu.com/p/416172020)



### 7.  A ***review*** of visual inertial odometry from filtering and optimisation perspectives（综述）

Jianjun Gui et al. “A review of visual inertial odometry from filtering and optimisation perspectives”. In: Advanced Robotics 29.20 (2015), 1289–1301. issn: 0169-1864. doi: {10.1080/01691864.2015.1057616}.

论文链接：[survey](https://www.tandfonline.com/doi/abs/10.1080/01691864.2015.1057616)

文献将VIO方法分成了两大类：基于优化；基于滤波。

基于优化的方法可以被视作最大似然估计 maximum likelihood；基于滤波的方法可以被视为最大后验估计 maximum a posterior

VIO中，IMU提供动态模型dynamic model，视觉提供测量模型measurement model

#### 基于滤波的方法

分为预测 prediction 和更新 updating 两步骤

通常使用EKF，PF

#### 基于优化的方法

分为**建图 mapping** 和**跟踪 tracking** 两步骤

基于优化的方法主要是**图像处理**的过程，IMU的测量数据通常作为 prior 或**正则项 regularisation term** 被加入到目标函数中。 mapping过程中，从图像中提取特征并计算重投影误差 reprojected error；tracking 过程中，利用优化方法计算移动平台的位姿变化。为了减小计算复杂度，还采用边缘化 marginalization 的方法。

优化方法中有一个图像配准 image alignment的过程，目的是根据图像，得到每一个时刻的位姿变换T； image alignment 可以分为 **feature alignment** 和 **dense alignment**。

feature alignment 需要先提取图像特征，然后计算重投影误差；

dense alignment 利用整幅图像的信息，计算光度误差 photometric error，常用算法是ICP

#### 优化方法和滤波方法的联系

IEKF等价于使用 Gauss-Newton的优化方法

#### Smoother

平滑器也利用了未来的信息

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220704145811756.png?token=AKAJBVJQV42CZRWJXQD4W7LDJD4B4" alt="image-20220704145811756" style="zoom:80%;" />

#### 边缘化和移动窗口估计

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220704145824676.png?token=AKAJBVPDB62ZO26SRYLW44DDJD4B6" alt="image-20220704145824676" style="zoom:80%;" />

#### 状态观测 state observability 和参数辨识 parameter identifiability



### 8. MSCKF

[A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation](https://intra.ece.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf)

Multi-state Constraint Kalman Filter

用EKF滤波方法做VIO

#### Motivation

之前传统的EKF-SLAM将***特征点（landmark）***加入到状态向量中与IMU的状态一起估计，因而当环境很大时，特征点很多，会产生极大的**计算复杂度**；

考虑到在移动过程中，相机位姿的个数远远小于特征点的数目，因此考虑将不同时刻的***相机位姿*加入到状态向量**；同一个特征点会被不同时刻的观察到，因而形成多个相机位姿间的几何约束（**geometric constraints**）。本文考虑用这样一个几何约束构建观测方程，进而建立ES-EKF

#### Method

状态向量（这里使用 error state）：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926165538537.png?token=AKAJBVO6KCSYVLVVG6A5W7DDJD4CA" alt="image-20220926165538537" style="zoom:50%;" />

状态更新方程：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926165618578.png?token=AKAJBVJUOURLSMEPCSBC4P3DJD4CC" alt="image-20220926165618578" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926165630856.png?token=AKAJBVI6EGBVWJ22REVQVJDDJD4CE" alt="image-20220926165630856" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926165639933.png?token=AKAJBVIFGLLD3JBJ6G6RPO3DJD4CG" alt="image-20220926165639933" style="zoom:50%;" />

观测方程的推导

首先想到，重投影误差（***实际观察到的特征点像素坐标***减去***估计的3D点投影到图像上的像素坐标***）：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926170056442.png?token=AKAJBVM2DNK2C4U4NXZSFFTDJD4CI" alt="image-20220926170056442" style="zoom:50%;" />

这样，经过线性化后能得到观测方程：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926170248939.png?token=AKAJBVISTDQDRUOGZ7OXUATDJD4CM" alt="image-20220926170248939" style="zoom:50%;" />

其中两个$\bold H$矩阵分别是观测$\bold z_i^{(j)}$对state和feature position的jacobian

这是不能用在EKF中的，因为还有与状态量无关的项$\bold H_{f_i}^{(j)G}\tilde{\bold p}_{f_j}$

论文中用的方法是：projecting $\bold r^{(j)}$ on the ***left nullspace*** of the matrix $\bold H_f^{(j)}$ 。做了个**零空间投影**，即对投影矩阵左乘$\bold A^T$，使得$\bold A^T\bold H_{f_i}^{(j)G}\tilde{\bold p}_{f_j}=0$，从而可以消去无关项，变为：

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20220926170819483.png?token=AKAJBVP3DRFIUNXAA2M23S3DJD4CO" alt="image-20220926170819483" style="zoom:50%;" />

上面方程只是对于单次观测，*每来一次新的观测*，就对状态量和观测方程做**augmentation**。然后可以得到最终的**观测方程**

MSCKF利用重投影约束作为观测模型，前提是需要知道特征点的3D坐标，这个3D坐标如何计算？

根据历史（EKF得到的）相机位姿和观测来**三角化**，从而计算**特征点的3D坐标**。为了保证三角化的精度，论文的做法是**特征点跟丢后**再三角化，保证利用到了所有历史信息。

***算法步骤***：

1. **IMU积分**：先利用[IMU加速度](https://www.zhihu.com/search?q=IMU加速度&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"76341809"})和角速度对状态向量中的IMU状态进行预测，一般会处理多帧IMU观测数据。
2. **相机状态扩增**：每来一张图片后，计算当前相机状态并加入到状态向量中, 同时扩充状态协方差.
3. **特征点三角化**：然后根据历史相机状态三角化估计3D特征点
4. **特征更新**：再利用特征点对多个历史相机状态的约束，来更新状态向量。注意：这里不只修正历史相机状态，因为历史相机状态和IMU状态直接也存在关系(相机与IMU的外参)，所以也会同时修正IMU状态。
5. **历史相机状态移除**：如果相机状态个数超过N，则剔除最老或最近的相机状态以及对应的协方差.



#### 不太了解的地方

* ES-EKF 误差EKF

  [Quaternion kinematics for the error-state Kalman filter](http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)

  为什么要用 error state 而不是 nominal state ？

  * 参数最少，参数量对应DOF，避免了 over-parametrization ；
  * 系统状态总处在原点附近，避免了可能的奇异性、gimbal lock 问题；
  * error state 更小，进而可以忽略雅可比计算中二阶小量，减小了计算复杂度；
  * error dynamic 更慢，KF的修正可以比预测频率更低

* 零空间投影

  > projecting $\bold r^{(j)}$ on the ***left nullspace*** of the matrix $\bold H_f^{(j)}$ 

  零空间投影是什么？

  [MSCKF的零空间投影](https://zhuanlan.zhihu.com/p/150641671)，就是**去除误差模型对三维点的依赖**

  什么是左零空间？

  矩阵的左零空间(**left nullspace**)定义:矩阵 M 的左零空间是所有满足$v^TM=0$ 的向量 v 的集合

  如何求$\bold A$？

  使用[givens rotation](https://zhuanlan.zhihu.com/p/136551885)求解方程$\bold A^T\bold H_{f_i}^{(j)G}\tilde{\bold p}_{f_j}=0$



### 9. LeGO-loam

[LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain](https://www.researchgate.net/profile/Tixiao-Shan/publication/330592017_LeGO-LOAM_Lightweight_and_Ground-Optimized_Lidar_Odometry_and_Mapping_on_Variable_Terrain/links/5d7c44a9a6fdcc2f0f6dc9e9/LeGO-LOAM-Lightweight-and-Ground-Optimized-Lidar-Odometry-and-Mapping-on-Variable-Terrain.pdf)

#### Motivation

传统的做法（LOAM）采用基于特征的方法做帧间匹配，通过计算点的曲率，提取出 edge 和 planar 特征。

但是，在UGV这样**算力**较低的平台下，LOAM的性能会恶化。同时，由于UGV所处环境的复杂性以及运动的非光滑性，激光雷达获得的数据往往会产生畸变，诸如树叶、草丛等景物会产生不可靠的特征，也就是说，需要一种方法能应对环境中的**噪声**。

#### Method

* Segmentation

  先将原始点云投影为**距离图 range image** ，然后用 column-wise evaluation **分离出 地面点**。对地面点之外的其他点，采用图像分割的方法对点云做**聚类**，点数小于30的类被忽略，从而去除 unreliable features

* Feature extraction

  类似loam，根据曲率做特征提取。

  从非地面点中提取出 edge *feature points* $\mathbb{F}_p$，从**非地面/地面点**中提取出 planar *feature points* $\mathbb{F}_e$；

  然后，从非地面点提取 edge *feature* $F_p$，从**地面点**提取出 planar *feature* $F_e$

* Lidar Odometry

  * 首先需要进行特征匹配。本文采用 label matching 的方法，只有属于**相同类别**的点和特征才会相互匹配。
  * Two-step L-M 优化：根据 planar 特征来做优化，得到$[t_z,\theta_{roll},\theta_{pitch}]$；然后对edge feature做优化，得到$[t_x,t_y,\theta_{yaw}]$

<img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221002181514414.png?token=AKAJBVMXDXGMW4ZGPNOZAG3DJD4CS" alt="image-20221002181514414" style="zoom:50%;" />

### 10. NeRF

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://dl.acm.org/doi/pdf/10.1145/3503250)



### 11. iMAP

[iMAP: Implicit Mapping and Positioning in Real-Time](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content/ICCV2021/papers/Sucar_iMAP_Implicit_Mapping_and_Positioning_in_Real-Time_ICCV_2021_paper.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=13502213069777937598&ei=gYg5Y7WyNI3MyQSdgLaoAw&scisig=AAGBfm3eWCskE4UFMcgdyD9CdlPQl1D8MQ)

[知乎iMAP解读](https://zhuanlan.zhihu.com/p/433672771)

<img src="https://pic3.zhimg.com/80/v2-bb51b03e3e8fb36ef5f929ec7ccd6256_1440w.webp" alt="img" style="zoom:50%;" />

#### Motivation

首个基于**RGB-D**相机，用**MLP**做场景表征**实时SLAM**系统

传统的**稠密建图**SLAM使用占用栅格图 occupancy map 或者 signed distance function 做场景表征，占用的**内存空间**很大；

相比于传统的TSDF场景表征方法，iMAP能**重建出未被观察到的地方**；

隐式场景表征 implicit scene representation 可以用来做相机位姿估计，但是都是离线形式，**计算需求量**大，本文使用深度图像能达到实时效果；

持续学习 continual learning 往往会遇到**灾难性遗忘 catastrophic forgetting** 的问题，本文采用 replay-based approach，将先前的结果缓存。

#### Method

分为两个线程，**Tracking** 固定网络（场景表征），优化当前帧对应的相机位姿；**Mapping** 对网络和选定的关键帧做联合优化。

* implicit scene neural network

  借鉴NeRF，用一个MLP将3D坐标点转换成颜色和体积密度值：
  $$
  \bold p=(x,y,z)\\
  F_\theta(\bold p)=(\bold c,\rho)
  $$
  但是与NeRF有区别，*没有考虑视角方向*（因为不需要对镜面反射建模）

  论文中还提到了 *Gaussian positional embedding* 将输入的3D坐标转换到n维空间（？）

* Depth and Color rendering

  通过 **query 得到的 scene network** 来从指定视角得到深度和颜色图像

  输入是相机位姿 $T_{WC}$和像素坐标$[u,v]$，先反投影到世界坐标系：
  $$
  \bold r=T_{WC}K^{-1}[u,v]
  $$
  然后在视角射线上采样：
  $$
  \bold p_i=d_i\bold r\qquad d_i\in\{d_1,\dots,d_N\}
  $$
  Query 之前得到的场景网络得到颜色和体积密度：
  $$
  (\bold c_i,\rho_i)=F_\theta(\bold p_i)
  $$
  将体积密度 **volume density** 转换为占据概率 **occupancy probability** ：
  $$
  \delta_i=d_{i+1}-d_i\\
  o_i=1-\exp(-\rho_i\delta_i)\\
  w_i=o_i\prod_{j=1}^{i-1}(1-o_j)
  $$
  得到深度和颜色：
  $$
  \hat{D}[u,v]=\sum^N_{i=1}w_id_i\\
  \hat{I}[u,v]=\sum^N_{i=1}w_i\bold c_i
  $$
  计算深度的方差：
  $$
  \hat{D}_{var}[u,v]=\sum^N_{i=1}w_i(\hat{D}[u,v]-d_i)^2
  $$

* joint optimisation

  用关键帧集合，针对**网络参数**和**相机位姿**做联合优化。使用ADAM优化器，使用的loss是**光度误差**和**几何误差**的加权

  光度误差 photometric error ：

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005203536327.png?token=AKAJBVJLLOFCEJN5DWHLTGLDJD4CS" alt="image-20221005203536327" style="zoom:50%;" />

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005203547885.png?token=AKAJBVLOC3K4K3NX5TJXPJDDJD4CU" alt="image-20221005203547885" style="zoom:50%;" />

  几何误差 geometric error，使用到了深度的方差 :

  <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005203632744.png?token=AKAJBVJH5CQLATD4IF2MBFDDJD4CW" alt="image-20221005203632744" style="zoom:50%;" />

  上面提到的都是 ***Mapping 阶段***

  ***Tracking 阶段***，与Mapping并行但是频率更高，针对 latest frame ，**固定场景网络**，使用之前提到的loss和优化器优化位姿

* Keyframe selection

  考虑到计算的复杂度和图像的冗余性（redundancy），基于 **information gain** 选取关键帧。

  **第一帧**一定会被选取，用来做网络初始化和世界坐标系的固定；

  每新增一个关键帧，保存一次network作为snapshot（之前提到的防止 catastrophic forgetting 的缓存机制）；

  后续关键帧的选取方法是，与**snapshot做比较**，看**是否观察到了新区域**。

  ***具体做法***：

  * 在 snapshot 和 frame 上随机选取 s 个像素点

  * 比较深度值差异，计算如下一个分数，然后与阈值做比较，小于就加入关键帧：

    <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005204534234.png?token=AKAJBVMKT2JLQAPYEI42NADDJD4CY" alt="image-20221005204534234" style="zoom:50%;" />

* **Active sampling**

  为了减少计算的复杂度，本文分别对像素点、关键帧都做了 Active Sampling

  * image active sampling

    这一步要对图像随机选点，但是要针对性地选择，更有价值（细节多、重建不够precise）的图像块要取更多的点。

    选取方法是先分块，先均匀采样，分块求loss，计算每一块的权重，然后再根据权重采样。

    <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005204937905.png?token=AKAJBVMEODG336PPLFRAPCLDJD4C4" alt="image-20221005204937905" style="zoom:50%;" />

  * keyframe active sampling

    为了限制联合优化的计算复杂度，每次迭代只选三个关键帧。同上，也是计算loss在分配权重

    <img src="https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221005205314858.png?token=AKAJBVIHRBBNRLGDWPZFPD3DJD4DA" alt="image-20221005205314858" style="zoom:50%;" />

#### Conclusion

MLP的结构收敛速度不够快，考虑结合***instant-NGP***？

[instant-NGP-repo](https://github.com/NVlabs/instant-ngp)

### 12. NICE-SLAM

[NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_NICE-SLAM_Neural_Implicit_Scalable_Encoding_for_SLAM_CVPR_2022_paper.pdf)

给定**RGB-D**图像序列，能在**大规模**（室内）场景下生成**密集几何**和准确的**相机跟踪**。

![image-20221014134614134](https://raw.githubusercontent.com/Yibeibankaishui/Notebook-pics/master/image-20221014134614134.png?token=AKAJBVIKBXSUBBYA3GIA2KTDJD4DG)

#### Motivation

本文提到对SLAM系统的要求是：

* real-time
* 能预测未被观察到的区域
* 能适用于 large scenes
* 对噪声足够 robust

神经隐式表达用作新视角图像合成等工作，通常是离线形式训练，给定相机位姿，训练时间长。

传统的SLAM方法，不能***预测未被观察到的区域***；learning-based 的方法，能够预测，对噪声和外点足够鲁棒，但是缺点是不 real-time，只能用于小场景。

与这篇工作最相关的是**iMAP**，iMAP输入是一个RGB-D图像序列，通过一个简单的MLP场景表征实现实时的稠密SLAM。iMAP由于单个MLP的结构，存在以下问题：

* 单个MLP由于容量有限，场景过大时，iMAP会失败
* 每次输入RGB-D帧都会更新整个MLP参数（**Global update**），因此会遇到 **Catastrophic forgetting** 的问题
* 收敛速度慢
* 生成的场景过于**smooth**，生成的场景几何不够detail，尤其在**大场景**下容易失效。

***NICE-SLAM*** 对场景表达进行改进，优化分层特征网格**（Feature grids + tiny MLPs）**，并结合不同空间层级下预训练的MLP的归纳偏差，从而在**大规模**场景下表现良好，解决了**网络遗忘**问题，同时达到了**更快的收敛速度**。

#### Method

本文使用层次化的场景表征 **hierarchical scene representation** ，从而使得SLAM系统能够囊括多层次的环境局部信息 **multi-level local information**。

![image-20221014140026214](文献阅读.assets/image-20221014140026214.png)

##### pipeline

* 前向传播（right-to-left）

  作为generative model，根据相机位姿和特征网格，渲染深度和RGB图像

* 反向传播（left-to-right）

  根据loss优化相机位姿和特征网络

  通过优化**特征网格**实现**Mapping**；优化**相机位姿**实现**Tracking**

##### 层次化场景表征

结合 **multi-level grid features** 和 **预训练的 MLP decoders** 做 occupancy probability 的预测，这是本文 geometry 的表征形式（替代简单MLP）；用一个 feature feature 和 无预训练的 decoder 表征 scene appearance ，即颜色。

* **Mid-level & Fine-level** Geometry representation

  以 **coarse-to-fine** 的方式，先优化 mid-level 的 feature grid，再用 fine-level 做 refinement

  * 首先，对于 mid-level 直接用预训练好的MLP解码出 occupancy

    <img src="文献阅读.assets/image-20221020143224854.png" alt="image-20221020143224854" style="zoom:50%;" />

    上式中，$\bold p$是3D点的位置，$\phi_\theta^1(\bold p)$是在 feature grid 中采用**三线性插值**方式得到的 **feature vector**，这里可以参考 ConvONet

  * 然后，以**残差**的方式添加 fine-level features。具体做法是，fine-level MLP 输入为 **mid 和 fine feature 的拼接**，输出为一个 residual ：

    <img src="文献阅读.assets/image-20221020143152510.png" alt="image-20221020143152510" style="zoom:50%;" />

  * 最终，一个点的occupancy：

    <img src="文献阅读.assets/image-20221020143204865.png" alt="image-20221020143204865" style="zoom:50%;" />

  > 问题：mid-level 和 fine-level 的MLP学习到的函数形式不太一样，具体是怎么分别预训练的？

* **Coarse-level** Geometry representation

  Coarse-level 的过程与 Mid-level 一样，但是 feature grid 的分辩率很低，与其他两个level分开优化。**目的是**用来捕捉 high-level 的场景几何信息，用来预测场景中**未被观察到的部分**

* **Color** representation

  与几何表征不同的是，这里**联合优化 解码器 **和颜色特征

  <img src="文献阅读.assets/image-20221020144103717.png" alt="image-20221020144103717" style="zoom:50%;" />

* ##### MLP预训练

  [ConvONet论文](https://arxiv.org/pdf/2003.04618.pdf)，另见 14.ConvONet

  本文使用了三个预训练的 MLP 来**将 grid features 解码为 occupancy values**。

  三个MLP decoder作为 ConvONet 的一部分被预训练。ConvONet 包含CNN编码器和MLP解码器，通过最小化 binary cross-entropy loss 来训练，然后本文只使用训练好的 MLP 作为 decoder。在此后的优化过程中，**MLP的参数固定**，通过**优化 features **来 fit the observation。

  

##### Color and depth rendering

* 由相机内外参计算 viewing direction 

* 在视角方向的 ray 上采样$N$个点：

  这里与iMAP不同的，**一**是 ***根据 depth image 做 importance sampling*** （相比于直接做stratified sampling , NeRF原文的做法），即在深度附近采样 $N_{imp}$个点，一共采样了 $N=N_{strat}+N_{imp}$个点；

  > stratified sampling 是什么？
  >
  > 体积渲染的分层采样（hierarchical volume sampling），通过更高效的采样策略减小估算积分式的计算开销，加快训练速度。
  >
  > [Nerf-知乎](https://zhuanlan.zhihu.com/p/390848839)

  **二**是直接用 **Occupancy probability** *instead of volume density*

* 根据采样的点，用分层次的 MLP decoder 计算 Occupancy 

* 计算 ray termination probability

* 计算 coarse depth, fine depth, color, depth variances

  <img src="文献阅读.assets/image-20221020133548635.png" alt="image-20221020133548635" style="zoom:50%;" />

##### Mapping

Mapping 阶段就是优化分层次的**场景表征**

在当前帧和被选帧中采样一共$M$个像素点，用 **staged optimization** 最小化 **geometric loss and photometric loss**。

loss的形式：

<img src="文献阅读.assets/image-20221020134602494.png" alt="image-20221020134602494" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221020134614865.png" alt="image-20221020134614865" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221020134630184.png" alt="image-20221020134630184" style="zoom:50%;" />

分阶段 **staged** 优化:

* 先用**几何误差** $\mathcal{L}_g^f$优化 **mid-level** feature grid $\phi_\theta^1$

* 用**几何误差** $\mathcal{L}_g^f$优化 **mid-level** feature grid $\phi_\theta^1$ 和 **fine-level** feature grid $\phi_\theta^2$（coarse-to-fine的过程）

* 以BA的方式，联合优化所有level的特征网格和 **color decoder**（注意，color decoder 并不是预训练好的，也要参与优化）

  <img src="文献阅读.assets/image-20221020135017210.png" alt="image-20221020135017210" style="zoom:50%;" />

> 这种分阶段优化有什么好处？
>
> This multi-stage optimization scheme leads to **better convergence** as the higher-resolution appearance and fine level features can ***rely on the already refined geometry coming from mid-level feature grid***

##### Tracking

Tracking 阶段负责优化当前帧的**相机位姿**，其过程与 Mapping 并行，采用带方差的 **modified** geometric loss 结合光度误差：

<img src="文献阅读.assets/image-20221020135119687.png" alt="image-20221020135119687" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221020135132783.png" alt="image-20221020135132783" style="zoom:50%;" />

> tracking也就是定位问题，传统的SLAM做法是什么？
>
> 这里仅仅结合这两个loss来优化位姿有什么不足？

##### 应对动态物体的trick

对像素进行过滤，直接去除 loss 极大的 pixel



##### Keyframe selection

与iMAP一样，这里也用了一个 global keyframe list，根据信息增益不断添加新的关键帧。但是有两个trick：

* 只选用与 current frame 有 visual overlap 的 keyframe 参与几何优化
* 只优化当前视锥内的 grid feature

<img src="文献阅读.assets/image-20221014172752239.png" alt="image-20221014172752239" style="zoom:35%;" />

> 这个做法的**好处**是：
>
> 保证能在 grid-based representation 的基础上做 **local updates**，从而避免了iMAP的遗忘问题
>
> 保证视野范围外的几何保持静止 ***ensures the geometry outside of the current view remains static*** ，并且由于只需要优化一部分参数，优化过程更efficient

#### ideas

只适用于室内场景；不能适用于更大规模的场景。对于更大规模场景的nerf可以参照mip-nerf, block-nerf

没有回环检测

### 13. BARF

[BARF : Bundle-Adjusting Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_BARF_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2021_paper.pdf)

本文提出了一种同时优化NeRF场景表征和相机位姿的方法。通过对NeRF的 positional encoding 做改进，采用 coarse-to-fine 的方式能同时优化得到高保真的场景表征和准确的相机位姿。

#### Motivation

原始的NeRF能生成 high fidelity 的新视角图像，但是前提是**相机位姿准确已知**。

本文想要**用不精确的相机位姿来训练NeRF**，即在**构建3D场景的同时做相机位姿配准**（*要解决的问题*）

#### Method

最naïve的想法是，*直接同时优化相机位姿和NeRF场景表征*。

本文通过对梯度的分析得出，NeRF中的 **positional encoding** 虽然能捕获高频特征、有助于生成**高保真的图像**；但是***对位姿的估计不利***。

positional encoding 的形式如下：

<img src="文献阅读.assets/image-20221023200729370.png" alt="image-20221023200729370" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221023200740286.png" alt="image-20221023200740286" style="zoom:50%;" />

同时优化场景表征和位姿的问题可表述为：

<img src="文献阅读.assets/image-20221023202443592.png" alt="image-20221023202443592" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221023202513332.png" alt="image-20221023202513332" style="zoom:50%;" />

对位姿 $\bold p$ 的优化使用梯度：
$$
\bold J(\bold u;\bold p)=\frac{\partial \mathcal{\hat I}}{\partial \bold p}=\sum^N_{i=1}\frac{\partial g(\bold y_1,\dots,\bold y_N)}{\partial \bold y_i}\frac{\partial \bold y_i(\bold p)}{\partial \bold x_i(\bold p)}\frac{\partial\mathcal{W}(z_i\bar{\bold u};\bold p)}{\partial\bold p}
$$
其中，$g$是组合所有采样点值的 volume rendering 函数，$\bold y=[\bold c;\sigma]^T=f(\bold x;\Theta)$是NeRF网络的输出即颜色和 volume density 的生成函数，$\mathcal{W}$是从相机坐标系到世界坐标系的坐标变换。

使用 positional encoding 之后，第二项梯度为
$$
\frac{\partial \bold y_i(\bold p)}{\partial \bold x_i(\bold p)}=\frac{\partial f(\bold x)}{\partial \bold x_i}=\frac{\partial f'\circ\gamma(\bold x)}{\partial \bold x_i}
$$
其中，第$k$个成分的雅可比为：

<img src="文献阅读.assets/image-20221023202657209.png" alt="image-20221023202657209" style="zoom:50%;" />

可见，随着$k$变大，梯度呈指数增长，这会导致梯度更新$\Delta\bold p$极不稳定

即，信号中的高频成分使得梯度不稳定，容易陷入 local minimum ，优化过程严重依赖初始值。如下图所示，光滑的信号有利于对 displacement 的预测。因此，论文的想法是用 **coarse-to-fine** 的做法，先在低频信号上做优化（相当于将图像/场景表征做 blur 处理），然后逐步扩大频率范围。

<img src="文献阅读.assets/image-20221023200156172.png" alt="image-20221023200156172" style="zoom:40%;" />

具体做法是，对 positional encoding 做改进，采用**带权重的编码**：

<img src="文献阅读.assets/image-20221023194426723.png" alt="image-20221023194426723" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221023194447580.png" alt="image-20221023194447580" style="zoom:50%;" />

权重参数$w$由$\alpha$控制，$\alpha\in[0,L]$随着迭代过程不断递增， positional encoding 向量中$k>a$的部分（即高频成分）被置零，不贡献梯度。这样，可以看成是对 positional encoding 加上了一个截止频率不断增长的动态低通滤波器，在优化过程中先优化低频部分，再优化高频部分，即 coarse-to-fine 过程。

#### Limitation

与NeRF本身的缺陷类似，优化和渲染慢，不能适用于大场景，并且没有引入时序和帧间 tracking。



### 14. ConvONet

[ConvONet论文](https://arxiv.org/pdf/2003.04618.pdf)

结合CNN和隐式场景表征，实现高质量**3D场景重建**。

#### Motivation

**要解决的问题**：3D重建，输入3D点云或者粗糙的3D voxel ，用Neural network恢复shape

<img src="文献阅读.assets/image-20221016160656398.png" alt="image-20221016160656398" style="zoom:28%;" />

论文提到***对3D场景表征的要求***：

* 能 encode 复杂几何和任意的拓扑结构
* 能适用于大场景
* 同时囊括 局部 和 全局 信息
* 能兼顾计算效率和占用内存空间

传统方法（诸如：Volumetric representation, Point clouds, Mesh-base representation）往往不能同时满足以上所有要求。

**隐式场景表征 deep implicit representation** 用学习到的 occupancy 或 符号距离函数（signed distance functions）来表征3D结构。但是之前的工作（Occupancy Network）存在问题：由于使用的是**简单的MLP**网络结构，不能整合**局部信息**（只有全局信息），也不能 *incorporating **inductive biases** such as **translation equivariance** into the model*，从而导致在**大场景**下失效，导致表面重建**过于平滑**。

#### Method

本文的 key idea 是，利用卷积操作（**CNN**）来得到 translation equivariance，同时利用 3D 结构的局部自相似性 *exploit the local self-similarity of 3D structures*

<img src="文献阅读.assets/image-20221014150555485.png" alt="image-20221014150555485" style="zoom:50%;" />

<img src="文献阅读.assets/image-20221016163904790.png" alt="image-20221016163904790" style="zoom:25%;" />

**Pipeline**: 

Encoder:

* 用 **PointNet** 提取点云特征，得到 local information 
* 将特征投影到 ground plane / 3 planes / voxel ；

* 用 **U-net** （2D U-net / three 2D U-net / 3D U-net）处理先前得到的特征，保证了 translation equaivariance， U-net 中的 skip connection 结构整合了全局和局部信息，最终得到 **feature grid**

Decoder:

* 给定点的3D位置，在 feature grid 中做 query，用线性插值得到 feature vector $\psi(\bold p,\bold x)$
* 联合$\psi(\bold p,\bold x),\bold p$ 输入MLP网络输出Occupancy Probability，即$f_\theta(\bold p,\psi(\bold p,\bold x))$

> ***对于Encoder，PointNet做了什么？提取到的特征是什么？***
>
> PointNet从无序点云中提取特征，用到3D目标识别和分割任务上。
>
> 提取到的特征 *Learns to summeraize a shape*, *key points forming the skeleton of an object*

#### ideas

主要关注NICE-slam和ConvONet的关联。

### 15. An Algorithm for the SE(3)-Transformation on Neural Implicit Maps for Remapping Functions

[An Algorithm for the SE(3)-Transformation on Neural Implicit Maps for Remapping Functions](https://arxiv.org/pdf/2206.08712.pdf)

为了应对NeRF场景表征中 non-remappable 的问题，提出了一种针对隐式场景表征的变换（transformation）方法。进而，可以解决基于NeRF的SLAM中的回环检测问题。



### 16. Instant NGP

[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)

NeRF的训练加速



### 17. PointNet

用深度学习做点云特征提取，从而用到识别和分割任务中。



### 18. LENS

[LENS: Localization enhanced by NeRF synthesis](https://proceedings.mlr.press/v164/moreau22a/moreau22a.pdf)

NeRF 辅助定位。使用NeRF合成的新视角图像来辅助机器人位姿估计



### 19. Mip-NeRF

[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2021/papers/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.pdf)

[project page](https://jonbarron.info/mipnerf/)

[知乎-mipNeRF](https://zhuanlan.zhihu.com/p/494514221)

NeRF只在相机位置固定、分辨率与训练图像一致的新视角生成上表现较好。当拉近、拉远时（在多分辨率下观察场景），图像会产生锯齿以及模糊。

> 混叠是什么？
>
> 根据奈奎斯特采样定理，当采样频率小于信号频率的2倍时，会出现混叠。此时，高频信号会被重构为低频信号
>
> NeRF的混叠，可以看成是，当在与训练图像不同的分辨率，尤其是更低分辨率下进行新视角图像生成时，相当于直接对场景表征做降采样，采样的频率低于场景中高频信息频率的两倍。

Mip-NeRF通过使用一个cone（圆锥）而不是ray采样，同时改进 positional encoding 为 integrated positional encoding （IPE）来克服混叠（aliasing）现象。当拉近、拉远相机时（在不同分辨率下观察图像），生成图像也有很好的效果。

> **Mip-NeRF introduces *low-pass filtering* over the Fourier features, where the filter size is controlled by the cone size**

#### Motivation

NeRF直接在射线上采样多个点，再进行 positional encoding ，随后作为MLP的输入。每个频率的信息以相同方式被直接编码（encodes all frequencies equally），从而导致生成图像中的高频成分会出现锯齿

<img src="文献阅读.assets/image-20221101210115234.png" alt="image-20221101210115234" style="zoom:40%;" />

用2D图像来类比：

如果直接降采样，效果很不好。相当于NeRF中直接采样点然后PE的过程

<img src="文献阅读.assets/image-20221101205849510.png" alt="image-20221101205849510" style="zoom:40%;" />

经过高斯滤波平滑操作后再降采样，图像质量提高了（相当于是过滤了高频成分）。因此Mip-NeRF的想法可以看作是先对场景表征做高斯滤波再降采样。

<img src="文献阅读.assets/image-20221101205912373.png" alt="image-20221101205912373" style="zoom:40%;" />

#### Method

<img src="文献阅读.assets/image-20221101210335339.png" alt="image-20221101210335339" style="zoom:30%;" />

Mip-NeRF用**圆锥（cone）**取样**射线**来进行采样。可以看出，NeRF的采样方式仅仅能体现一个极小点的特征，即使是以不同方向来采样，对同一个点来说，其特征也不变，具有歧义性（**ambiguity**）；而Mip-NeRF的做法考虑到了特征的**形状和大小**，是对锥台（conical frustram）内一块体积的特征进行建模 ***models the volume of each sampled conical frustum***，从而去除了歧义性。

NeRF由于其采样和编码方式，只能学习到特定scale的特征，所以需要使用 coarse 和 fine **两个**等级的MLP；而Mip-NeRF本身建模出的特征就**包含尺度**信息，所以***仅用一个MLP***，这使得模型大小减半。

<img src="文献阅读.assets/image-20221101210511970.png" alt="image-20221101210511970" style="zoom:30%;" />



Mip-NeRF 使用 IPE/integrated positional encoding 来表征 conical frustum 中 volume 的特征

具体而言，是用锥台中所有点的 positional encoding 的期望来作为锥台 volume 内的特征表征。如果直接进行计算，就是如下形式：

<img src="文献阅读.assets/image-20221101214820170.png" alt="image-20221101214820170" style="zoom:45%;" />

<img src="文献阅读.assets/image-20221101214839821.png" alt="image-20221101214839821" style="zoom:45%;" />

式中，分子不好计算。因此论文采用多元高斯分布来近似。由于圆锥台关于射线对称的形状，仅需要三个参数就可以表征这个高斯分布。即在射线上的距离均值$\mu_t$，射线方向上的方差$\sigma_t$，垂直于射线方向的方差$\sigma_r$

<img src="文献阅读.assets/image-20221101214937735.png" alt="image-20221101214937735" style="zoom:45%;" />

然后，进行相对坐标系到世界坐标系的坐标转换：

<img src="文献阅读.assets/image-20221102003246890.png" alt="image-20221102003246890" style="zoom:80%;" />

进行重新参数化（reparameterization），再利用高斯分布的线性变换：

<img src="文献阅读.assets/image-20221102004223965.png" alt="image-20221102004223965" style="zoom:80%;" />

<img src="文献阅读.assets/image-20221102004242436.png" alt="image-20221102004242436" style="zoom:80%;" />

由高斯分布经三角函数变换后的均值：

<img src="文献阅读.assets/image-20221102005226160.png" alt="image-20221102005226160" style="zoom:80%;" />

根据线性性质，得到IPE：

<img src="文献阅读.assets/image-20221101215204479.png" alt="image-20221101215204479" style="zoom:45%;" />

对于$\boldsymbol\Sigma_\gamma$，由于PE向量的各个维度相互独立，因此文中的做法是只算对角：

<img src="文献阅读.assets/image-20221102005508004.png" alt="image-20221102005508004" style="zoom:80%;" />

<img src="文献阅读.assets/image-20221102005519075.png" alt="image-20221102005519075" style="zoom:80%;" />

这样，类比于2D图像中的**高斯滤波**，NeRF场景表征中的高频信息相当于是被平滑掉了，所以达到了 anti-aliasing 的效果。

> **In short, IPE preserves frequencies that are constant over an interval and softly “removes” frequencies that vary over an interval, while PE preserves all frequencies up to some manuallytuned hyperparameter L**

### 20. Block-NeRF

[Block-NeRF: Scalable Large Scene Neural View Synthesis](https://openaccess.thecvf.com/content/CVPR2022/papers/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.pdf)

[youtube](https://www.youtube.com/watch?v=sJbCaWMaDx8)

[project website](https://waymo.com/intl/zh-cn/research/block-nerf/)

**大规模场景**的NeRF。

本文将大场景划分为多个block，每个block训练单独的NeRF网络。在 inference 阶段，比如在特定位置和视角下，Block-NeRF能无缝结合相关的小的NeRFs来生成图像

#### Motivation

先前的工作尽管可以生成高质量的新视角图像，但是大部分都注重于小尺度、物体级别的重建，并不能适用于城市中的大尺度场景。

#### Method

单个NeRF不能适用于大尺度的城市场景。本文将环境分成数个Block-NeRF，每个Block-NeRF在**训练阶段**被分别并行训练；在**推理阶段**，选定几个Block-NeRF组合以生成图像。

这种做法使得场景表征**可以被扩展**（添加格外的Block-NeRF），并且可以通过仅改变一个或几个Block-NeRF而*不用重新训练整个场景*的方式来不断更新。

<img src="文献阅读.assets/image-20221102193959994.png" alt="image-20221102193959994" style="zoom:50%;" />

NeRF网络结构上，借鉴了**Mip-NeRF**；相机位姿的优化借鉴了**BARF**的方法；另外使用了**NeRF in the wild**中的外观嵌入。

##### Block Size and Placement

论文中的方式是，选择在道路交叉口放置每个Block-NeRF，能覆盖临近道路段75%的距离，同时保证相邻Block-NeRF有50%的重合。

> 个人想法：这个场景完全是网格化的街区，如果街区形状不规则怎么办？

##### 单个Block-NeRF的训练 Training Individual Block-NeRFs

* Appearance Embeddings

  借鉴**NeRF-W**（*NeRF in the wild: Neural Radiance Fields for Unconstrained Photo Collections*），使用Generative Latent Optimization来优化外观嵌入向量。**Appearance Embedding** 使得网络能够表**征不同的光照和天气状况**，在此基础上可以采用插值的方式来控制外观生成不同光照、天气状况下的环境图像

* Learned Pose Refinement

  将Pose与场景表征进行**联合优化**

* Exposure Input

  考虑相机**曝光**的因素，将表征曝光的参数加入到外观预测部分中

* Transient Objects

  对于动态的物体，本文首先**假设**场景几何在训练数据中具有连贯性（consistent），然后使用一个**语义分割模型**对明显违反假设的动态物体生成mask，训练过程中忽略那一部分

* Visibility Prediction

  用了一个额外的小的MLP来学习被采样点的**能见度（visibility）**，其输入是点的位置和视角方向，输出是透射率（transmittance）$T_i$，表式从特定的相机角度去观察一个点的能见度。如果完全不可见，输出应该为0，否则为1，或者介于0-1之间。

  这个MLP网络$f_v$由预测density的$f_\sigma$**提供监督信息**，透射率在$f_\sigma$中的计算式为：

  <img src="文献阅读.assets/image-20221102194338088.png" alt="image-20221102194338088" style="zoom:50%;" />

##### 多个Block-NeRF的合并 Merging Multiple Block-NeRFs

* Block-NeRF Selection

  对于给定的视角，只渲染与之相关的blocks。具体做法是考虑**距离和能见度**，设定距离半径，与视角位置太远的block不考虑；另外，能见度visibility低于设定阈值的也被去除。

* Block-NeRF Compositing

  使用前一阶段经滤除而选定的blocks，根据逆距离权重（**inverse distance weighting**）来进行插值（interpolate），从而生成特定视角下的图像。

* Appearance Matching

  之前所述的**外观嵌入**（这一段用的是appearance latent code，应该与appearance embedding等价）在训练阶段被随机初始化，这将导致在不同的block-NeRF中，code相同，外观却不同。

  为了解决这个问题，需要对code进行优化。具体做法是，先根据能见度，选定与毗邻的几个block-NeRFs有关的特定的3D位置（3D matching location），然后**固定NeRF网络参数**，用${l}_2$损失来***优化 appearance latent code***，在这一过程中，block的外观会不断变化。

#### Limitations

* 应对动态物体的问题
* 计算复杂度问题

### 21. Structure-Aware NeRF without Posed Camera via Epipolar Constraint

[Structure-Aware NeRF without Posed Camera via Epipolar Constraint](https://arxiv.org/pdf/2210.00183.pdf)





### 22.CLONeR: Camera-Lidar Fusion for Occupancy Grid-aided Neural Representations

[CLONeR: Camera-Lidar Fusion for Occupancy Grid-aided Neural Representations](https://arxiv.org/pdf/2209.01194.pdf)





### 23.  iNeRF

[iNeRF](https://arxiv.org/pdf/2012.05877.pdf)

使用梯度下降法，最小化当前位姿下NeRF生成的图像和被观察图像的像素差，来优化相机位姿估计





### 24. Loc-NeRF: Monte Carlo Localization using Neural Radiance Fields

[Loc-NeRF](https://arxiv.org/pdf/2209.09050.pdf)

结合蒙特卡洛算法做位姿估计，用NeRF表征场景地图。优势是不需要依赖初始位姿估计并且比已有的NeRF算法更快



### 25. NeRF in the wild: Neural Radiance Fields for Unconstrained Photo Collections

[NeRF in the wild](https://openaccess.thecvf.com/content/CVPR2021/papers/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.pdf)





### 26.Dense Depth Priors for Neural Radiance Fields from Sparse Input Views

[Dense Depth Priors for Neural Radiance Fields from Sparse Input Views](https://openaccess.thecvf.com/content/CVPR2022/papers/Roessle_Dense_Depth_Priors_for_Neural_Radiance_Fields_From_Sparse_Input_CVPR_2022_paper.pdf)



### 27. GARF

[Gaussian Activated Neural Radiance Fields for High Fidelity Reconstruction & Pose Estimation](https://arxiv.org/pdf/2204.05735.pdf)

GARF在BARF的基础上，提出了使用 Gaussian activations 的 ***positional embedding-free*** 的神经辐射场结构，能在提供高保真重建的同时估计准确的相机位姿



### 28. NeRF-SLAM

[NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields](https://arxiv.org/pdf/2210.13641.pdf)



### 29. Orbeez-SLAM

[Orbeez-SLAM: A Real-time Monocular Visual SLAM with ORB Features and NeRF-realized Mapping](https://arxiv.org/pdf/2209.13274.pdf)



### 30. iDF-SLAM

[iDF-SLAM: End-to-End RGB-D SLAM with Neural Implicit Mapping and Deep Feature Tracking](https://arxiv.org/pdf/2209.07919.pdf)



### 31. NeDDF

[NEURAL DENSITY-DISTANCE FIELDS](https://arxiv.org/pdf/2207.14455.pdf)



### 32. SDF-based RGB-D Camera Tracking in Neural Scene Representations

[SDF-based RGB-D Camera Tracking in Neural Scene Representations](https://arxiv.org/pdf/2205.02079.pdf)



### TODO

<img src="文献阅读.assets/image-20221024150646657.png" alt="image-20221024150646657" style="zoom:40%;" />

* GARF
* Nerfies
* Stru

---

## 泡泡机器人





## Zachary Teed - Optimization Inspired Neural Networks for Multiview 3D

### limitation of optimization methods

设计目标函数、优化的难点

<img src="文献阅读.assets/image-20221023155746837.png" alt="image-20221023155746837" style="zoom:50%;" />

more expressive objective functions + stronger priors : more difficult optimization problems

### Deep learning for Multiview 3D

1. for Depth prediction
2. Optical Flow : FlowNet
3. Visual Odometry : DeepVO
4. SfM : DeMoN

存在的问题：训练困难；难以在不同数据集中泛化

### 
