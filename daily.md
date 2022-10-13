本周主要在结合矩阵分析的教科书，调研和学习这一系列工作中矩阵的推导。

learning 验证了这类矩阵反映了一些线性变换DFT，DCT，……，1DConv的结构。
Kaleidoscope的结论是反映在算法电路中深度不深的矩阵都属于这种矩阵，

10.10 

在尝试链接 vscode 到机器上的时候出现了一些链接上的问题，

10.11
近几天在尝试build fly的镜像，由于电脑cuda是11.7而包里是11.3，尝试通过WSL Ubuntu来做。

密钥问题：
https://www.cnblogs.com/2205254761qq/p/11863928.html

sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 76F1A20FF987672F

GCC 版本问题：
https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version

10.12

https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa

ubuntu 里还是有 cuda 相关的问题，我尝试回到本机和集群上 build 镜像。

docker build -t yuzy_torch1.10.1_py38_cu11.3_fly . 

yuzy_torch1.10.1_py38_cu11.3_fly

我注意到集群上有

luoxin_py3.8_pytorch1.10.1_cu11.3_devel_lighnting_hydra_mmcv_wandb_timm_v2

该镜像，可能可以使用。


