import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste
from siim import SIIM
from transforms import Resizeit
from cnn_embedding import CNNEmbedder

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            #criterion = nn.CrossEntropyLoss()
            pos_weight  = 32542 / 584
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float)
            criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight)
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit':
        from vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    elif args.model_name == 'vit_emb':
        from vit_embedded import ViTEmbedded
        args.in_c = 3
        args.num_classes=2
        args.size = 8
        args.padding = 4
        net = ViTEmbedded(args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=8, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token)
    elif args.model_name == 'cnn':
        hparams = {
            'batch_size': 8,
            'learning_rate': 1e-2,
            'epochs': 1
        }
        net = CNNEmbedder(hparams=hparams)
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    # train_transform += [
    #     transforms.RandomCrop(size=args.size, padding=args.padding)
    # ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")

    if args.dataset == 'siim':
        train_transform += [transforms.Resize(size=(32, 32))] 
        test_transform += [transforms.Resize(size=(32, 32))] 
    else:
        train_transform.append(transforms.ToTensor())
        test_transform.append(transforms.ToTensor())


    train_transform += [
        #transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    # if args.rcpaste:
    #     train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        #transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)  
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)
    
    elif args.dataset == "siim":
        args.in_c = 3
        args.num_classes=2
        args.size = 32
        args.padding = 4
        #root = f'{root}/siim'
        root = f'/home/ra49tad2/cv_attention/data/siim'
        args.mean, args.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        train_transform, test_transform = get_transform(args)
        
        train_ds = SIIM(root, purpose='train', seed=args.seed, split=0.7, transforms=train_transform)#, tfm_on_patch=tfm_on_patch)
        test_ds = SIIM(root, purpose='val', seed=args.seed, split=0.7, transforms=test_transform)#, tfm_on_patch=tfm_on_patch)
       # train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        #test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
