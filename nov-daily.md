python main.py /gpub/imagenet_raw --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

已经知道了 P 的写法，接下来要在 test_s2d 中进行测试：

首先，新建一个新 mlp model，state dict deepcopy

用 re.match 找到所有 blkdiag 层的名称，两个一组提取出来，将参数作为 w1w2bfly 输入函数，输出一个 fc1.weight 的形式

然后， 新 state dict 读对应的 monarch 的剩余部分，和 fc2.weight 乘

这一部分的代码已经写好了，接下来需要调试的是主程序，



对于 S2D 的实验设计，我们需要将 epoch 分为两部分，首先要求初始模型是 monarch，

第一部分设置 70% 的 epoch，对使用了 monarch 的模型进行训练，然后新建模型，读 statedict 并处理。

继续进行剩下的 30 % 训练。

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 mymain.py /workspace/imagenet --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp --s2d --replace-mlp --epochs 10 --s2ddebug

用 dataloader 的优势，比如说原来 dataloader 很慢，那用上 flash attention 就能体现加速

维护文档，整理一些可行的加速方法

accelerated deep learning

用 gpu 的东西做一个 encoder/decoder, 图形无损格式，怎么用 GPU 读得更高效，编码再反编码，编码之后的部分更高效。 butterfly 能不能做一个更高级的 dataloader 的方式？

再到算法，用 butterfly，sparse，decentralized trainning

model ，数据，bp，单卡场景，分布式场景（通讯） 数据密集

模型密集，每个模型加一定的网络结构

要知道每个模块哪里可以做加速

写两个readme/github/google doc rebuttal，总之配个文档把这个东西写清楚

单机单卡 节约计算，单机多卡，多机多卡 节约通信

我们第一阶段先做单机的

后面，可以考虑写成一本 book/survey/tutorial 的形式，一边读文章一边去想，添新的东西

从训练 pipeline 的角度, 有 engineer trick，有

矩阵稀疏分块，比如说是 n^2 ， 不知是 diagnal block上是dense 的不重要，关键是找到数值比较大的区域，在这些地方做泛化，现在问题是只保留了 diagnal 上的块，但其实可以考虑其他地方的块，可以考虑每个块里是 sparse。数值小不代表 cor 小。

目前是数值小就去掉，思考一下矩阵乘的特点，qkv 上也可以做很多加速，有可能是 Lowrank+Sparse。

关键是找到数值比较大的区域，在这些地方做泛化，比如说把矩阵分成

11.5

tao 说，矩阵近似的工作关键是找到数值比较大的区域，在这些地方做泛化，而 Monarch 的工作把矩阵分成块对角结构，是不是只保留了对角线上的块，其实可以考虑其他地方的块，比如说将矩阵分成方阵块而不是对角线块，每个块里面是稀疏矩阵这样的形式。因为分成对角线块可能将小数值抹去，但数值小不代表 cor 小。需要思考一下矩阵乘的特点。

Attention 方法 QKV 上也可以做加速，有可能是 LowRank+Sparse 的形式。

沈老师好，刚刚陶老师打电话来讨论现在的工作，提了两点，

第一点是Monarch 的工作是不是保留块对角，其他元素置零。

我回应是只保留了块对角，这样可以利用 PyTorch 的矩阵乘 bmm 计算块的乘积，

他说，矩阵近似的工作关键是找到数值比较大的区域，在这些地方做泛化，而 Monarch 的工作把矩阵分成块对角结构，是不是只保留了对角线上的块，其实可以考虑其他地方的块，比如说将矩阵分成方阵块而不是对角线块，每个块里面是稀疏矩阵这样的形式。分成对角线块是否是把小数值抹去了，但数值小不代表 cor 小，需要思考一下矩阵乘的特点。

这个我没有马上回应，现在考虑一下感觉 Monarch 的应该没有问题，因为从原始矩阵到 Monarch 的转换本质上是一个降维处理，是矩阵元素重新排列，划分成方阵块进行 SVD，每块保留最大的那个特征值对应的两个向量，最终形成两个块对角矩阵乘积的形式，这里面抹去的是小特征值而不是原始矩阵里小的数值。

第二点是 Attention 方法 QKV 上也可以做加速，这个可以考虑，有可能是 LowRank+Sparse 的形式。

我说 butterfly 的作者也做了这个的工作，这个应该就是 Pixelated 那篇。

陶老师好，前面您电话里说 Monarch 的工作把矩阵分成块对角结构，是不是只保留了对角线上的块，将小数值抹去了，

我刚刚仔细思考了一下，Butterfly 矩阵考虑的不只是对角块，包含的是类似图里这样一组对角线的稀疏结构，Monarch 是这一组矩阵乘积最后泛化才形成块对角线的形式。


从原始矩阵到 Monarch 的转换本质上是做降维处理去掉小特征值，是原始矩阵的每个元素重新排列后划分成方阵块，进行 SVD，M = U Σ V^T 每个块矩阵块保留最大的那个特征值对应的两个 U, V 向量，重新排列形成 Monarch 里面两个块对角矩阵乘积的形式。

您说的进一步思考矩阵乘中 block 之间的 correlation 很有道理，我之后仔细考虑