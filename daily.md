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

startdocker -u "-it" -D /gpub/imagenet_raw -c /bin/bash bit:5000/yuzy_torch1.10.1_py38_cu11.3_fly

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

lowrank+sparse 有用，但涉及到改动 attention 的结构，和 monarch 和 flashattention 不一样，所以目前先跑后面两个。cuda 非常重要。

现在分布式这一块属于基础、硬核的工作

先说结果，再展示过程。