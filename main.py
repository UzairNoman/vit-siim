
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from net import Net
class Settings:
    def __init__(self):
        self.dataset = "ham"
        self.num_classes = 7
        self.model_name = "vit"
        self.patch = 8
        self.batch_size = 128
        self.eval_batch_size = 1024
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_epochs = 2
        self.weight_decay = 5e-5
        self.warmup_epoch = 2
        self.precision = 16
        self.criterion = "ce"
        self.smoothing = 0.1
        self.dropout = 0.0
        self.head = 12
        self.num_layers = 7#12
        self.hidden = 384#768#
        self.label_smoothing = False
        self.mlp_hidden = 384#3072
        self.seed = 42
        self.project_name = "VisionTransformer"
        self.off_benchmark = False
        self.dry_run = False
        self.autoaugment = False
        self.rcpaste = False
        self.cutmix = False
        self.mixup = False
        self.off_cls_token = False
        self.api_key = False

args = Settings()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")
train_ds, test_ds = get_dataset(args)
# classes = torch.tensor([0, 1, 2])
# indices = (torch.tensor(train_ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
# train_ds = torch.utils.data.Subset(train_ds, indices)
# test_ds = torch.utils.data.Subset(test_ds, indices)
#print(data.shape)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    if args.api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args.api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with TB")
        # logger = pl.loggers.CSVLogger(
        #     save_dir="logs",
        #     name=experiment_name
        # )
        logger = TensorBoardLogger(name=experiment_name,save_dir="logs")
        refresh_rate = 1
    setting_attrs = vars(args)
    print(', '.join("%s: %s" % item for item in setting_attrs.items()))
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark,logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)