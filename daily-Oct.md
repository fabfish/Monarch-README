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

bit:5000/nvidia-cuda11.3.1-cudnn8-devel-ubuntu20.04

该镜像，可能可以使用。

需要在本地准备好github库，然后迁移到远程，并放到docker中安装

scp -P 22 .\Miniconda-latest-Linux-x86_64.sh yuzy@192.168.9.99:~/dockertmp/

尝试conda安装的时候发现速度非常慢，明天解决。可以考虑用pip，参考malf-cuda101-torch13

yuzy_torch1.10.1_py38_cu11.3_fly

--network host
    --build-arg HTTP_PROXY=http://192.168.16.5:3128
    --build-arg HTTPS_PROXY=http://192.168.16.5:3128

scp -P 39099 file_name yuzy@202.38.69.241:~/dockertmp/

https://www.jianshu.com/p/9a4d1b4db99a
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

目前已经完成镜像

yuzy_torch1.10.1_py38_cu11.3_fly

10.13 尝试使用该镜像，对fly进行测试

下次编译镜像的时候可以加入自己喜欢的 oh-my-zsh，之后可以记得看看。

为了在远程修改代码，添加 git proxy

export http_proxy=http://192.168.16.5:3128
export https_proxy=http://192.168.16.5:3128

git config --global http.proxy http://192.168.16.5:3128



调试流程：

先准备好代码

scp -P 39099 -r .\fly\ yuzy@202.38.69.241:~/ 

进入镜像
ssh G101

startdocker -u "-it" -c /bin/bash bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

需要挂载数据目录 

-D /gpub/imagenet_raw

startdocker -u "-it" -D /gpub/imagenet_raw -c /bin/bash bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

zsh 2

python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=full datamodule.data_dir=/gpub/imagenet_raw eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 81.7% acc

python run.py experiment=imagenet/mixer/mixerb-cutmix-fbbflylr datamodule.data_dir=/path/to/imagenet model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.butterfly_size=8 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.n_factors=2 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.block=32 

HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/mixer/mixerb-cutmix-fbbflylr datamodule.data_dir=/gpub/imagenet_raw model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.butterfly_size=8 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.n_factors=2 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.block=32 

这边可能由于docker内hydra的配置问题，找不到对应的模块。
可能是因为在hydra配置文件里没有加hydraworkdir。

考虑先研究vitae的代码，把vitae跑起来。后面不行的话就直接迁移butterfly层到vitae上。

python 3.7
timm 0.4.12

可以用这个安装缺少的库

python3 -m pip install --no-index --no-build-isolation -e ./lightning-bolts

用 pip 安装几个没装上的包的 wheel

cd /ghome/yuzy/
pip install tqdm-4.64.1-py2.py3-none-any.whl

要等一下

pip install pytorch_lightning_bolts-0.2.5-py3-none-any.whl

另一个 torchmetrics 也需要安装
https://pytorch-lightning.readthedocs.io/en/1.4.4/extensions/metrics.html

hydra里不是模块缺失，而是config文件错误。
下面这句被转义了，去掉$相关的
sed -i 's/batch_size_eval: ${eval:${.batch_size} * 2}/# batch_size_eval: ${eval:${.batch_size} * 2}/g' /ghome/yuzy/fly/configs/datamodule/imagenet.yaml

文件修改相关的都可以通过 terminal 来完成了。

10.14 总结

```bash

echo $CUDA_VISIBLE_DEVICES

nvidia-smi

export CUDA_VISIBLE_DEVICES="0,1,..."

!!!!!!!只能在G101改

startdocker -u "-it --ipc=host" -D /gpub/imagenet_raw -c /bin/zsh bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

zsh 2

cd /ghome/yuzy/

pip install tqdm-4.64.1-py2.py3-none-any.whl pytorch_lightning_bolts-0.2.5-py3-none-any.whl torchmetrics-0.10.0-py3-none-any.whl pytorch_block_sparse-0.1.2.tar.gz

export PATH=$PATH:/home/user/.local/bin

sed -i 's/from pytorch_lightning.metrics.functional import accuracy/from torchmetrics.functional import accuracy/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/models/regression/logistic_regression.py

sed -i 's/from torch._six import container_abcs, string_classes/from torch._six import string_classes\nimport collections.abc as container_abcs/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/datamodules/async_dataloader.py

cd fly

HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/mixer/mixerb-cutmix-fbbflylr datamodule.data_dir=/gpub/imagenet_raw model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.butterfly_size=8 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.n_factors=2 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.block=32
```

sed -i 's/# Build the model/# Build the model\n    print("##################\nkwargs:\n",kwargs,"##################\nmodel_cfg:\n",model_cfg,"##################\n")/g' /home/user/conda/lib/python3.8/site-packages/timm/models/helpers.py

修了一个报错：timm default_cfg 被重命名为 pretrained_cfg

注释掉 mixers.yaml 里的 # - override /logger: wandb（容器没网络连接），devices 改成 1

https://github.com/pytorch/pytorch/issues/67864

目前还在排队

10.15
vitae

python validate.py /gpub/imagenet_raw --model ViTAE_basic_Tiny --eval_checkpoint ./che

/ghome/yuzy/ViTAE-Transformer/Image-Classification/vitae/checkpoints

vit-fly 结构的研究：

vit 中，transformer 的各个 block 包含 attention 和 mlp 层，都可以用指定的 butterfly 层替代。

可用的 butterfly sparse 有

讨论：

lowrank+sparse 有用，但涉及到改动 attention 的结构，和 monarch 和 flashattention 不一样，所以目前先跑后面两个。cuda 非常重要。

现在分布式这一块属于基础、硬核的工作

先说结果，再展示过程。

HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/vit/vit-s-butterflyblockdiag-mlp datamodule.data_dir=/gpub/imagenet_raw

昨天没跑通 sanity check 是因为没加 --ipc==host 导致内存不够，目前设置多卡还有问题，以及不清楚是否需要另外更改 monarch 的配置。

[yuzy@gwork ~]$ qsub myjob_vit_normal.pbs
380963.Ghead
[yuzy@gwork ~]$ qsub myjob_vit_monarch.pbs
380964.Ghead

10.16 

今天任务是把 monarch 配置到 vitae 上，首先需要把 vitae 配好，

考虑到之前运行出现问题，打算重建算法库，删除之前修改过的文件。

timm 0.4.12

```bash

startdocker -u "--ipc=host -it" -D /gpub/imagenet_raw -c /bin/zsh bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

cd /ghome/yuzy

pip install 

python validate.py /gpub/imagenet_raw --model ViTAE_basic_Tiny --eval_checkpoint ./checkpoints

```

vitae 中的问题：
1. dataset 无法 import
cannot import name 'Dataset' from 'timm.data'
2. 无 real_json
在 https://github.com/google-research/reassessed-imagenet 下载一个
3. 
checkpoints:

python validate.py /gpub/imagenet_raw --model ViTAE_basic_Tiny --eval_checkpoint ./checkpoints/ViTAE-T.pth.tar

切换到timm 0.4.12，validate可以正常工作

python -m torch.distributed.launch --nproc_per_node=1 main.py /gpub/imagenet_raw --model ViTAE_basic_Tiny -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

python -m torch.distributed.launch --nproc_per_node=1 main.py /gpub/imagenet_raw --model ViTAE_basic_Tiny -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

把节点数量换成 1，可以跑通。可能 timm 0.4.12 也可以适合 fly 的库，但 fly 的代码我改了一下。

monarch: in 256, out 512, nblocks=4, in blocks=64, out blocks=128

blkdiag1 torch.Size([4, 64, 64]) n, insize, insize
blkdiag2 torch.Size([4, 128, 64])n, outsize,insize

normcell input: [64,196,256]
norm2 [64,196,256]

mlp is MonarchMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

MonarchMlp(infeature = 256, hidden_features = 512)


Monarch forwarding, x shape is  torch.Size([128, 197, 768])
Monarch forwarding, x shape is  torch.Size([128, 197, 2304])




Train: 0 [   0/10009 (  0%)]  Loss:  6.926911 (6.9269)  Time: 23.170s,    5.52/s  (23.170s,    5.52/s)  LR: 1.000e-06  Data: 8.405 (8.405)
Train: 0 [  50/10009 (  0%)]  Loss:  6.927876 (6.9180)  Time: 1.462s,   87.53/s  (1.896s,   67.51/s)  LR: 1.000e-06  Data: 0.026 (0.199)
Train: 0 [ 100/10009 (  1%)]  Loss:  6.911202 (6.9174)  Time: 1.413s,   90.60/s  (1.647s,   77.73/s)  LR: 1.000e-06  Data: 0.027 (0.112)

Train: 0 [   0/10009 (  0%)]  Loss:  6.917073 (6.9171)  Time: 24.811s,    5.16/s  (24.811s,    5.16/s)  LR: 1.000e-06  Data: 5.169 (5.169)
Train: 0 [  50/10009 (  0%)]  Loss:  6.920621 (6.9153)  Time: 1.435s,   89.20/s  (1.985s,   64.48/s)  LR: 1.000e-06  Data: 0.016 (0.142)
Train: 0 [ 100/10009 (  1%)]  Loss:  6.906080 (6.9162)  Time: 1.439s,   88.93/s  (1.717s,   74.56/s)  LR: 1.000e-06  Data: 0.016 (0.079)


沈老师好，我刚刚调好了一个仅替换 NormalCell 中 mlp 的 vitae-monarch，
对于 Vitae-basic-Tiny 模型，
可训练的参数可以从 4882792（mlp）减少到 3735912（monarch），
但训练速度和准确度现在排不上队，暂时不知道结果，
训练 100 个 epoch 时，粗略的速度是（1080 卡，batch size 128，imagenet size=224，lr 0.001）
batch_time 1.647s, rate 77.73/s （mlp）
batch_time 1.717s, rate 74.56/s （monarch）
速度基本接近，理想状况应该是准确度差不多，速度提升，我考虑这个速度可能是刚开始训练，或者模型很小导致的

[yuzy@gwork ~]$ qsub my_vitaes_mlp.sh
381128.Ghead
[yuzy@gwork ~]$ qsub my_vitaes_monarch.sh
381129.Ghead

10.17

ssh yuzhiyuan11@jdea-cq-jump.jd.com -p 80

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

docker run -it -v /public/data0/DATA-1/users/yuzhiyuan11:/workspace --name yuzhiyuan11 hpcaitech/cuda-conda:11.3 /bin/bash

ssh jdops1003@172.17.226.143

cd /public/data0/DATA-1/users/yuzhiyuan1

docker run -it -v /public/data0/DATA-1/users/yuzhiyuan11:/workspace --name yuzhiyuan11 hpcaitech/cuda-conda:11.3 /bin/bash

docker exec -it yuzhiyuan11 /bin/bash

10.19

网络问题，目前先暂时采用 --network=host 的方式，之后再考虑更安全的修改（改网络之类的

对于报错：ATen/cuda/CUDAGraphsUtils.cuh: No such file or directory

先按正常方式安装 pytorch

cutlass/cutlass.h: No such file or directory

在 src 下找到 cutlass 路径，删掉该文件夹，替换为 git clone cutluss

cuda 报错
RuntimeError: CUDA out of memory. 
Tried to allocate 2.00 GiB 
(GPU 0; 39.59 GiB total capacity; 
13.50 GiB already allocated; 
1.76 GiB free; 
15.63 GiB reserved in total by PyTorch) 
If reserved memory is >> allocated memory 
try setting max_split_size_mb to avoiement and PYTORCH_CUDA_ALLOC_CONF

出现这个问题可能是因为现在有比较大的实验在跑，

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

[root@DGX-107 flash-attention]# PYTHONPATH=$PWD python benchmarks/benchmark_flash_attention.py
FlashAttention - Forward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd76ee0>
fn(*inputs, **kwinputs)
  4.52 ms
  1 measurement, 30 runs , 128 threads
FlashAttention - Backward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd765b0>
y.backward(grad, retain_graph=True)
  9.11 ms
  1 measurement, 30 runs , 128 threads
FlashAttention - Forward + Backward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd76fa0>
f(grad, *inputs, **kwinputs)
  12.98 ms
  1 measurement, 30 runs , 128 threads
PyTorch Standard Attention - Forward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd76760>
fn(*inputs, **kwinputs)
  19.25 ms
  1 measurement, 30 runs , 128 threads
PyTorch Standard Attention - Backward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd76730>
y.backward(grad, retain_graph=True)
  47.09 ms
  1 measurement, 30 runs , 128 threads
PyTorch Standard Attention - Forward + Backward pass
<torch.utils.benchmark.utils.common.Measurement object at 0x7f996dd768b0>
f(grad, *inputs, **kwinputs)
  67.52 ms
  1 measurement, 30 runs , 128 threads


ms        flash_attn  pytorch
Forward   4.52        19.25    
Backward  9.11        47.09
F+B       12.98       67.52

10.22

conv 层的加速


机器学习，cuda

docker run -idt --name ubuntu-cuda-113 --gpus all --shm-size 128g -v /public/data0/DATA-1/users/chenshixiang6:/workdir --dns 114.114.114.114 --dns 8.8.8.8 --dns 114.114.114.114 --dns 8.8.8.8 nvidia/cuda:11.3.0-devel-ubuntu16.04

10.29

1. 确定 monarch S2D 的用法，并部署，最迟这两天完成

2. 配置 flash-attention-vit

3. 整理 monarch butterfly 系列的模型细节，写论文要用。

没有找到 monarch S2D 的实现，找到的是 blockdiag (not butterfly)，是否等价于 monarch？（形式上等价）

在 experiment/bert 下，bertlarge-blockdiag-densified，它将会在训练的第二部分调用 warmup，读取 state dict 后，将 checkpoints 里面的


但不是初始化一组权值矩阵（nblock,out_feature/nblock, in_feature/nblock）训练，理论上来讲

目前找到的并不是butterfly block diag, 而只是简单的 block diag bert 的配置文件。

相关的文件分别是：
configs\
  experiment\
    bert\
      bertlarge-blockdiag-densified.yaml  我们感兴趣的实验，在训练后期将块对角转化为实心矩阵
      bertlarge.yaml 配置实验，在 model 中选择 bertlarge

  model\
    bert_mlp_cfg/blockdiag.yaml 设置 bert large 的 bert layer 中的 mlp 为 blockdiag
    bertmodel\bertlarge.yaml 实验配置

src\
  models\
    layers\
      blockdiag_butterfly_multiply  我们希望使用的 bflym，传入 x 与两个块对角矩阵，通过 einsum 等方式实现乘法
  ops\
    blockdiag_multiply.py 实际使用到的 bm，就是将块矩阵填补成实心的
  utils\
    checkpoint.py 中有一个 blockdiag to dense mlp bert 函数，就是读 statedict，处理 encoder 中的 bb mlp 层

实验设置是，当 bert large 训练时，用 nblock = 4 块对角矩阵来代替 mlp，进行稀疏训练；然后填补成实心矩阵，进行后半段训练。

通过读 checkpoints，将 diagblock_mlp 读取出来，展开填补成 dense

那么用于 vitae 上，且使用 fly，我们应该做的是：

找到训练函数，重新设置 epoch 为 70%，当训练时，用 monarch linear，先训练 70% epoch

然后新写一个读 checkpoints 的函数，要搜索出改过的 monarch 层，参照 densify 的写法读 statedict，读完之后新写一个 convert 函数来转换，目前的思路是参照 blockdiag_butterfly_multiply_reference 中的 version 3， return out2? 或者就按照定义做乘法，先写一个 test 试试，参照孙研学长的做法

11.2

现在，先把 10 月 20 号训好的 checkpoint 读取出来，分别是这两个

回顾时注意，这些代码是在 gdata1 上存储的

ViTAE-Transformer\Image-Classification\output\train\20221019-201710-ViTAE_basic_Small-224
ViTAE-Transformer\Image-Classification\output\train\20221020-233150-ViTAE_basic_Small-224

startdocker -u "--ipc=host -it" -P /gdata1/yuzy -D /gpub/imagenet_raw -c /bin/zsh bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

<!-- python mymain.py /workspace/imagenet --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp -->

下面是 mlp 的

python mymain.py /gpub/imagenet_raw --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp --resume /gdata1/yuzy/ViTAE-Transformer/Image-Classification/output/train/20221020-233150-ViTAE_basic_Small-224/last.pth.tar

下面是 monarch 的

python mymain.py /gpub/imagenet_raw --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp --resume /gdata1/yuzy/ViTAE-Transformer/Image-Classification/output/train/20221019-201710-ViTAE_basic_Small-224/last.pth.tar
