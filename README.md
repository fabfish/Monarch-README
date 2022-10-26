这是关于 https://github.com/HazyResearch/fly 的配置文档。

库中包含以下几篇论文中的工作：

Monarch: Expressive Structured Matrices for Efficient and Accurate Training （https://arxiv.org/abs/2204.00595）

Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models （https://arxiv.org/abs/2112.00029）

下面介绍如何在集群上测试库中 Monarch 相关的内容，这些在 d107（A100）上测试过。

使用 git clone 下载仓库，根目录有一个作者提供的 Dockerfile 可以用于 docker image build，