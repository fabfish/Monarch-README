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