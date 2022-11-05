# 简介

这里是关于将 https://github.com/HazyResearch/fly 移植到 https://github.com/ViTAE-Transformer/ViTAE-Transformer 的配置文档，本文档介绍如何配置使用 monarch 替换 mlp 的 vitae 实验。

我们使用 fly 库中包含的以下论文中的工作：

Monarch: Expressive Structured Matrices for Efficient and Accurate Training （https://arxiv.org/abs/2204.00595）

和 vitae 库中包含于以下论文中的工作：

ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias（https://arxiv.org/abs/2106.03348）

# 如何使用

下面介绍如何在集群上配置实验，这些在 d107（A100）上测试过。

## 下载仓库

为了运行我们的实验，原 vitae 仓库 image-classification 需要修改，需要修改的地方是：

- 添加 imagenet 的 real.json 文件。
    - 在 https://github.com/google-research/reassessed-imagenet 下载了一个
- 报错：cannot import name 'Dataset' from 'timm.data'
    - 这是由于 timm 的版本问题，0.4.12 可以解决
- 添加 monarch 相关的配置文件
- 添加 checkpoints

改的地方比较多了，推荐使用个人修改好的仓库：

```bash
git clone https://github.com/yourid/fly
```

该仓库已经修改好，可以直接使用，大小为 90 MiB。

## 启动镜像

docker 镜像已经在 DGX-107 上准备好，可以直接使用。关于镜像准备，请参考附录“准备镜像”。

```bash
# first-time use
nvidia-docker run -it -v /public/data0/DATA-1/users/yuzhiyuan11:/workspace -v /public/data0/DATA-1/users/mipeng7/datasets/imagenet2012:/workspace/imagenet --name yuzhiyuan11-fly --network=host --dns 114.114.114.114 --dns 8.8.8.8 --shm-size 64g nvcr.io/nvidia/pytorch:22.06-py3-fly /bin/bash

# second-time use
docker restart yuzhiyuan11-fly
docker attach yuzhiyuan11-fly
```

## 运行代码

下面的代码可以用于 ViTAE-Small + Monarch 的测试。

由于 timm 版本更新，旧版参数中的 default_cfg 被 pretrained_cfg 替代，导致 unexpected keyword argument 报错，因此先将 image 中预置的 timm==0.6.5 退回 0.4.12 版本。

```bash
pip install timm==0.4.12

cd ViTAE-Transformer/Image-Classification/

# test vitae + monarch
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 main.py /workspace/imagenet --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

# to test vitae + mlp, modify ViTAE-Transformer/Image-Classification/vitae/NormalCell.py
```

如果需要测试 ViTAE-Small + Mlp，将 ViTAE-Transformer/Image-Classification/vitae/NormalCell.py 第 121 行的注释去掉，将第 122 行注释。后续会通过参数命令修改得更规范一些。

```python
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = MonarchMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
```

# 代码结构

接下来简单介绍为了使用 Monarch 而对 vitae 仓库做的修改。目前采用了简单直接的方法，将 fly 库中的 src 文件夹全部放在 Image-Classification 的 vitae 目录下。在 vit-monarch 文档中我们介绍过：

---

与 vit 实验设置中使用 monarch 层有关的主要 config 文件是 /fly/configs/experiment/imagenet/vit/vit-s-butterflyblockdiag-mlp.yaml

它将以 configs/model/vitmlp_cfg/butterflyblockdiag.yaml 中的配置设置一组实验，

后者将会修改 vit 中使用的 mlp 的类型，也就是在 src/models/layers/mlp.py 的 MlpCustom 函数中使用 src/models/layers/monarch_linear.py 中的 MonarchLinear 替换 Mlp 中的结构。

这将会被 vit.py 中的 VisionTransformer 使用，VisionTransformer 用 Block 定义一个 Attention + mlp 结构，Block 定义于 src.models.modules.vision_common 中，而 mlp 的配置将被传递到这里，用于更换原定的 mlp 层。 

---

在 vitae 目录下，我们模仿 mlp.py 中的 MlpCustom，将其中的 Linear 层换成 MonarchLinear，便完成了修改。由于 vitae 没有使用 dotenv 库，我们向相关的文件中都添加了 import 路径前的 . 符号。

# 附录

## 准备 Docker 镜像

docker 镜像已经在 DGX-107 上准备好，可以直接使用，注意回退 timm 的版本至 0.4.12。