
import torch
import pytorch_lightning as pl
import numpy as np
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
from pytorch_lightning.loggers import TensorBoardLogger
from net import Net
import time

class Settings:
    def __init__(self):
        self.dataset = "c100"
        self.num_classes = 100
        self.criterion = "ce"

        self.model_name = "coat"
        self.batch_size = 128
        self.eval_batch_size = 32
        self.max_epochs =  30

        self.patch = 8
        self.num_layers = 8#12
        self.hidden = 384#768#
        self.mlp_hidden = 1536#384#3072
        self.head = 12

        self.lr = 1e-3
        self.min_lr = 1e-5
        self.label_smoothing = False
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 5e-5
        self.warmup_epoch = 5
        self.precision = 16
        self.smoothing = 0.1
        self.dropout = 0.0
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
train_ds, val_ds, test_ds = get_dataset(args)
# classes = torch.tensor([0, 1, 2])
# indices = (torch.tensor(train_ds.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
# train_ds = torch.utils.data.Subset(train_ds, indices)
# test_ds = torch.utils.data.Subset(test_ds, indices)
#print(data.shape)

# cnn needed drop last
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)

# print("val",next(iter(val_dl)).shape)
# print(next(iter(test_dl)).shape)

if __name__ == "__main__":
    t0 = time.time()
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
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=val_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)
    if not args.dataset == "siim":
        trainer.test(dataloaders=test_dl)

    t1 = time.time()
    exec_time = (t1 -t0)/3600
    print(f'{exec_time} hours')
