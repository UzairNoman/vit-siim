import os
import torchvision.transforms as transforms
import pandas as pd
from siim import SIIM
import torch
from net import Net
import numpy as np
import torchvision
from utils import get_model, get_dataset, get_experiment_name, get_criterion
import time
from tqdm import tqdm
class Settings:
    def __init__(self):
        self.dataset = "c10"
        self.num_classes = 10
        self.criterion = "ce"

        self.model_name = "vit"
        self.batch_size = 128
        self.eval_batch_size = 1024
        self.max_epochs = 30

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
        
if __name__ == "__main__":
    args = Settings()
    t0 = time.time()
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

    data_root = "/dss/dsshome1/lxc09/ra49tad2/ViT-CIFAR/data"
    # df_test = pd.read_csv(os.path.join(data_root, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_root, 'test', f'{x}.jpg'))
    n_test = 8
    # repo_root = os.path.abspath(os.getcwd())
    # data_root = os.path.join(repo_root, "data/siim")
    test_transform = []
    # test_transform += [transforms.Resize(size=(224, 224))]
    test_transform = transforms.Compose(test_transform)


    args.in_c = 3
    args.size = 32
    test_ds = torchvision.datasets.CIFAR10(data_root, train=False, transform=test_transform, download=False)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)
    print(len(test_ds))
    OUTPUTS = []
    model = Net(args)
    args.experiment_name = f"{args.model_name}_{args.dataset}"

    model_path = f"weights/{args.experiment_name}.pth"
    print(model_path)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    print(f'Model loaded: {model_path}')


    sample_submission = pd.read_csv('../input/cifar-10/sampleSubmission.csv')
    train_labels = pd.read_csv('../input/cifar-10/trainLabels.csv')


    def predict_for_submit():
        """
        Predict submission test data and form dataframe to submit
        """
        print("Forming submission dataframe...")
        
        # Predict
        y_pred = model.predict(test_ds)
        y_pred = np.argmax(y_pred, axis=1)
       
        sample_submission.label = y_pred
    
        print(f"Submission dataframe created. Rows:{len(sample_submission.label.values)}")
        
        # Write to csv
        sample_submission.to_csv('submission.csv', index=False)
        print("Submission completed: written submission.csv")
        return sample_submission

    
    sample_submission = predict_for_submit()

    # PROBS = []
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with torch.no_grad():
    #     for data in tqdm(test_loader):         
    #         data = data#.to(device)
    #         probs = torch.zeros((data.shape[0], 10)).to(device)
    #         for I in range(n_test):
    #             l = model(data).to(device)
    #             probs += l.softmax(1)
    #         probs /= n_test
    #         PROBS.append(probs.detach().cpu())

    # PROBS = torch.cat(PROBS).numpy()

    # OUTPUTS.append(PROBS[:, 1])


    # pred = np.zeros(OUTPUTS[0].shape[0])
    # for probs in OUTPUTS:
    #     pred += pd.Series(probs).rank(pct=True).values
    # pred /= len(OUTPUTS)

    # df_test['target'] = pred
    # df_test[['image_name', 'target']].to_csv(f'submission-{args.experiment_name}.csv', index=False)

    t1 = time.time()
    exec_time = (t1 -t0)/60
    print(f'{exec_time} mins')
