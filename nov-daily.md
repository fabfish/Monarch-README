python main.py /gpub/imagenet_raw --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

已经知道了 P 的写法，接下来要在 test_s2d 中进行测试：

首先，新建一个新 mlp model，state dict deepcopy

用 re.match 找到所有 blkdiag 层的名称，两个一组提取出来，将参数作为 w1w2bfly 输入函数，输出一个 fc1.weight 的形式

然后， 新 state dict 读对应的 monarch 的剩余部分，和 fc2.weight