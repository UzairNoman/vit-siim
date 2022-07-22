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
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(hparams)
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
        
        auc_score = metrics.roc_auc_score(label.cpu(), out[:, 1].cpu().squeeze().detach().numpy())
        self.log('auc', auc_score, on_step=True, on_epoch=True)
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

        auc_score = metrics.roc_auc_score(label.cpu(), out[:, 1].cpu().squeeze().detach().numpy())
        self.log('auc', auc_score, on_step=True, on_epoch=True)
        val_acc = torchmetrics.functional.accuracy(out[:, 1], label)
        self.log('valid_acc_from_tmet', val_acc, on_step=True, on_epoch=True)
        self.log('valid_acc', acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss,on_step=True, on_epoch=True)

        fpr, tpr, thresholds = roc_curve(label.cpu(), out[:, 1].cpu())
        auc_rf = auc(fpr, tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format(auc_rf,self.hparams.model_name))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        self.logger.experiment.add_figure('AUC Curve', plt.gcf(), self.current_epoch)

        return { 'loss': loss.item(), 'preds': out, 'target': label}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=self.hparams.num_classes)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(self.hparams.num_classes), columns=range(self.hparams.num_classes))
        plt.figure(figsize = (self.hparams.num_classes,self.hparams.num_classes*2))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

        fpr, tpr, thresholds = roc_curve(targets.cpu(), preds[:, 1].cpu())
        auc_rf = auc(fpr, tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format(auc_rf,self.hparams.model_name))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC/AUC curve')
        plt.legend(loc='best')
        self.logger.experiment.add_figure('ROC/AUC Curve', plt.gcf(), self.current_epoch)

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