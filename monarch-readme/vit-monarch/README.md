# 简介

这里是关于 https://github.com/HazyResearch/fly 的配置文档，本文档介绍如何配置使用 monarch 替换 mlp 的 vit 实验。

fly 库中包含以下几篇论文中的工作：

Monarch: Expressive Structured Matrices for Efficient and Accurate Training （https://arxiv.org/abs/2204.00595）

Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models （https://arxiv.org/abs/2112.00029）

# 如何使用

下面介绍如何在集群上测试库中 Monarch 相关的内容，这些在 d107（A100）上测试过。

## 下载仓库

由于代码运行使用 wandb 作为 logger，在运行时会调用 git 验证仓库的 ownership，因此目前个人找到的最好的办法是 fork 一份作者的代码（https://github.com/HazyResearch/fly）。正好为了运行我们的实验，原 fly 仓库代码有几处需要修改，我们可以在这时将其修改，需要修改的地方是：

- ImageNet 处理中有一处会报缺少参数错误（原因可能是某个库如 lightning 版本更新了，导致参数变化）。
- Hydra logger 配置会因冲突无法复制 run.py 文件，导致不能正常运行训练。
    - 见 https://github.com/explosion/spaCy/discussions/10097

修改方法如下：

在 GitHub 上 fork 作者的代码（页面右上角）后，找到 /fly/src/datamodules/imagenet.py，做以下修改：

在第 38 行的 \_\_init\_\_ 函数中，在 self 后另起新行，向 args 添加 

```python
        mixup,
        batch_size_eval,
```

在第 67 行 super().\_\_init\_\_(*args, **kwargs) 后，添加

```python
        self.mixup = mixup
        self.batch_size_eval = batch_size_eval
```

找到 configs/experiment/imagenet/t2tvit/t2tvit7.yaml，做以下修改：

在第 23 行，把 devices: 8 改成 devices: 1。这是为了避免前述的 hydra 冲突，目前没有找到更好的办法。

```yaml
  devices: 1
```

然后进入工作目录，使用 git clone 下载仓库。测试时（10 月 26 日） dgx-107 网络有点波动，可能网络不太好，需要等一会儿。如果卡住可以重试几次。

```bash
git clone https://github.com/yourid/fly
```

## 启动镜像

docker 镜像已经在 DGX-107 上准备好，可以直接使用。如果容器 monarch-test 已经 stop，可以直接 restart 该容器并使用。关于镜像准备，请参考附录“准备镜像”。

从镜像新建容器（挂载 mipeng 的数据集），以及已有容器使用的指令分别是：

```bash
# first-time use
nvidia-docker run -it -v /public/data0/DATA-1/users/{yourname}:/workspace -v /public/data0/DATA-1/users/mipeng7/datasets/imagenet2012:/workspace/imagenet --name monarch-test --network=host --dns 114.114.114.114 --dns 8.8.8.8  nvcr.io/nvidia/pytorch:22.06-py3-fly /bin/bash

# second-time use
docker restart monarch-test
docker attach monarch-test
```

## 运行代码

首先下载 checkpoints（速度可能比较慢），然后进行实验，代码在下面给出。

如果下载很慢，可以 mkdir -p checkpoints/t2tvit 后使用 cp 命令复制 yuzhiyuan11 目录下的文件，绝对路径位于 /public/data0/DATA-1/users/yuzhiyuan11/fly/checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar.og

训练中需要配置 wandb 账号。

```bash
# download checkpoints
cd fly/
mkdir -p checkpoints/t2tvit
cd checkpoints/t2tvit
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar

cd ../../
python scripts/convert_checkpoint_t2t_vit.py checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar

# test vit-small + monarch, code exec at /fly/ 
HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/vit/vit-s-butterflyblockdiag-mlp datamodule.data_dir=/workspace/imagenet

# test vit-small + mlp
HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/vit/vit-s datamodule.data_dir=/workspace/imagenet
```

两则 python 指令都在 /fly 目录中运行，前面加 HYDRA_FULL_ERROR=1 以方便 debug 使用，实验时可以删去。


# 代码结构

接下来简单介绍 fly 仓库中与 Monarch 相关的代码。代码使用了 hydra 框架，通过 yaml 文件的配置来更改。

与 vit 实验设置中使用 monarch 层有关的主要 config 文件是 /fly/configs/experiment/imagenet/vit/vit-s-butterflyblockdiag-mlp.yaml

它将以 configs/model/vitmlp_cfg/butterflyblockdiag.yaml 中的配置设置一组实验，

后者将会修改 vit 中使用的 mlp 的类型，也就是在 src/models/layers/mlp.py 的 MlpCustom 函数中使用 src/models/layers/monarch_linear.py 中的 MonarchLinear 替换 Mlp 中的结构。

这将会被 vit.py 中的 VisionTransformer 使用，VisionTransformer 用 Block 定义一个 Attention + mlp 结构，Block 定义于 src.models.modules.vision_common 中，而 mlp 的配置将被传递到这里，用于更换原定的 mlp 层。 


# 附录

## 准备 Docker 镜像

docker 镜像已经在 DGX-107 上准备好，可以直接使用。

下面分别记录编译作者提供的镜像，以及由自建镜像安装各种依赖包的方法：

### 作者提供的镜像

根目录有一个作者提供的 Dockerfile 可以用于编译 docker 镜像，为了方便编译，我们新建一个 dockertmp 文件夹，用于存储 Dockerfile 和可能会用到的文件，接下来将 Dockerfile 复制进去，然后进行镜像编译。

在公司集群上，作者提供的 Dockerfile 需要做些修改，包括添加 DNS 以便下载内容。

```bash
cd fly

mkdir dockertmp

cd dockertmp/

vim Dockerfile
```

为节约篇幅，修改过的 Dockerfile 文件放在本文档同目录下，可以将其中的内容复制，粘贴进 Dockerfile， esc:wq， 然后执行

```
docker build .
```

其中 fast-transformer 的 setup 需要较长时间，可能半个小时。


### 自建镜像（未完成）

自建镜像可以方便我们使用 Flash-Attention 仓库

```bash
# 启动 docker，挂载 imagenet
nvidia-docker run -it -v /public/data0/DATA-1/users/yuzhiyuan11:/workspace -v /public/data0/DATA-1/users/mipeng7/datasets/imagenet2012:/workspace/imagenet --name yuzhiyuan11-monarch --network=host hpcaitech/cuda-conda:11.3 /bin/bash
# 更换清华源
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda config --set show_channel_urls yes
vim ~/.condarc
```

把清华源贴进去

```.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

```
conda clean -i
conda install -y -c pytorch cudatoolkit=11.3 pytorch=1.10.1 torchvision torchtext
```

```
sh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/g' /workspace/.zshrc
```