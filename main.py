
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
from pytorch_lightning.loggers import TensorBoardLogger
class Settings:
    def __init__(self):
        self.dataset = "siim"
        self.num_classes = 2
        self.model_name = "vit"
        self.patch = 8
        self.batch_size = 128
        self.eval_batch_size = 1024
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_epochs = 8
        self.weight_decay = 5e-5
        self.warmup_epoch = 5
        self.precision = 16
        self.criterion = "ce"
        self.smoothing = 0.1
        self.dropout = 0.0
        self.head = 12
        self.num_layers = 7
        self.hidden = 384
        self.label_smoothing = False
        self.mlp_hidden = 384
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

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out[:,1], label.float())

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            #self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        auc = metrics.roc_auc_score(label, out[:, 1].squeeze())
        self.log('auc', auc, on_step=True, on_epoch=True)
        self.log('acc', acc, on_step=True, on_epoch=True)
        self.log('loss', loss,on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        #self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
        self.log("lr", self.optimizer.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out[:,1], label.float())
        acc = torch.eq(out.argmax(-1), label).float().mean()
        #self.log("val_loss", loss)
        #self.log("val_acc", acc)

        auc = metrics.roc_auc_score(label, out[:, 1].squeeze())
        self.log('auc', auc, on_step=True, on_epoch=True)
        val_acc = torchmetrics.functional.accuracy(out[:, 1], label)
        self.log('valid_acc_from_tmet', val_acc, on_step=True, on_epoch=True)
        self.log('valid_acc', acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss,on_step=True, on_epoch=True)


        return { 'loss': loss.item(), 'preds': out, 'target': label}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=args.num_classes)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(args.num_classes), columns=range(args.num_classes))
        plt.figure(figsize = (args.num_classes,args.num_classes*2))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        # repo_root = os.path.abspath(os.getcwd())
        # data_root = os.path.join(repo_root, "logs")
        # list_of_files = glob.glob(f'{data_root}/*') # * means all if need specific format then *.csv
        # latest_file = max(list_of_files, key=os.path.getctime)
        # writer = SummaryWriter(latest_file)
        # writer.add_figure("Confusion matrix", fig_, self.current_epoch)

    # def _log_image(self, image):
    #     grid = torchvision.utils.make_grid(image, nrow=4)
    #     self.logger.experiment.log_image(grid.permute(1,2,0))
    #     print("[INFO] LOG IMAGE!!!")

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
        print("[INFO] Log with CSV")
        # logger = pl.loggers.CSVLogger(
        #     save_dir="logs",
        #     name=experiment_name
        # )
        logger = TensorBoardLogger(name="vit_siim",save_dir="logs")
        refresh_rate = 1
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark,logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)