#!/bin/bash

cd /ghome/yuzy/

pip install tqdm-4.64.1-py2.py3-none-any.whl pytorch_lightning_bolts-0.2.5-py3-none-any.whl torchmetrics-0.10.0-py3-none-any.whl pytorch_block_sparse-0.1.2.tar.gz

export PATH=$PATH:/home/user/.local/bin

sed -i 's/from pytorch_lightning.metrics.functional import accuracy/from torchmetrics.functional import accuracy/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/models/regression/logistic_regression.py

sed -i 's/from torch._six import container_abcs, string_classes/from torch._six import string_classes\nimport collections.abc as container_abcs/g' /home/user/.local/lib/python3.8/site-packages/pl_bolts/datamodules/async_dataloader.py

cd fly

HYDRA_FULL_ERROR=1 python run.py experiment=imagenet/vit/vit-s datamodule.data_dir=/gpub/imagenet_raw