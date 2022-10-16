#!/bin/bash

cd /ghome/yuzy/

pip install timm-0.4.12-py3-none-any.whl

export PATH=$PATH:/home/user/.local/bin

sed -i 's/from pytorch_lightning.metrics.functional import accuracy/from torchmetrics.functional import accuracy/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/models/regression/logistic_regression.py

sed -i 's/from torch._six import container_abcs, string_classes/from torch._six import string_classes\nimport collections.abc as container_abcs/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/datamodules/async_dataloader.py

cd ./ViTAE-Transformer/Image-Classification

sed -i 's/# self.mlp = Mlp/self.mlp = Mlp/g' /ghome/yuzy/ViTAE-Transformer/Image-Classification/vitae/NormalCell.py 

python main.py /gpub/imagenet_raw --model ViTAE_basic_Small -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp

sed -i 's/self.mlp = Mlp/# self.mlp = Mlp/g' /ghome/yuzy/ViTAE-Transformer/Image-Classification/vitae/NormalCell.py 