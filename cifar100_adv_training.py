import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import os
import math
import time
import sys
from tqdm import tqdm
import argparse
from logger import get_logger

LOGGER = get_logger("cifar100_adv_training")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFAR100Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(50.0 / n_repeats))))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    LOGGER.info(f"Learning rate set to {lr}")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def fgsm(gradz, step_size):
    """Fast Gradient Sign Method"""
    return step_size * torch.sign(gradz)


def save_checkpoint(state, is_best, save_dir="./checkpoints"):
    """Save checkpoint and best model"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, f'checkpoint_epoch{state["epoch"]}.pth.tar')
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, "model_best.pth.tar")
        torch.save(state, best_filepath)
        LOGGER.info(f"Saved best model with accuracy {state['best_prec1']:.2f}%")


def train(train_loader, model, criterion, optimizer, epoch, config, global_noise_data):
    # Initialize meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to GPU
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        for j in range(config["n_repeats"]):
            # Ascend on the global noise
            noise_batch = Variable(
                global_noise_data[0 : input.size(0)], requires_grad=True
            ).cuda()

            # Add noise to input
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(MEAN).div_(STD)
            # Forward pass
            output, _ = model(in1)

            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, config["fgsm_step"])
            global_noise_data[0 : input.size(0)] += pert.data
            global_noise_data.clamp_(-config["clip_eps"], config["clip_eps"])

            # Update weights
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            LOGGER.info(
                "Train Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    top1=top1,
                    top5=top5,
                    cls_loss=losses,
                )
            )
            sys.stdout.flush()

    return top1.avg


def validate_pgd(val_loader, model, criterion, config):
    """Validate with PGD"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    eps = config["clip_eps"]
    step = config["fgsm_step"]

    # Switch to evaluation mode
    model.eval()
    for K in [20, 100]:
        for i, (input, target) in tqdm(enumerate(val_loader)):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            orig_input = input.clone()
            randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
            input += randn
            input.clamp_(0, 1.0)
            for _ in range(K):
                invar = Variable(input, requires_grad=True)
                in1 = invar - MEAN
                in1.div_(STD)
                output, _ = model(in1)
                ascend_loss = criterion(output, target)
                ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
                pert = fgsm(ascend_grad, step)
                # Apply purturbation
                input += pert.data
                input = torch.max(orig_input - eps, input)
                input = torch.min(orig_input + eps, input)
                input.clamp_(0, 1.0)

            input.sub_(MEAN).div_(STD)
            with torch.no_grad():
                # compute output
                output, _ = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                end = time.time()
                batch_time.update(time.time() - end)

                if i % 50 == 0:
                    print(
                        "PGD Test: [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            i,
                            len(val_loader),
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )
                    sys.stdout.flush()

        print(
            " PGD-{K} Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
                K=K, top1=top1, top5=top5
            )
        )


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Normalize input
            input = input.sub_(MEAN).div_(STD)
            # Forward pass
            output, _ = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                LOGGER.info(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                sys.stdout.flush()

    LOGGER.info(
        " * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
    )
    return top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 Adversarial Training for Free"
    )
    parser.add_argument(
        "--data-path",
        default="/path/to/cifar100",
        type=str,
        help="path to CIFAR-100 data",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--batch-size", default=128, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--weight-decay", default=0.0002, type=float, help="weight decay"
    )
    parser.add_argument(
        "--n-repeats",
        default=4,
        type=int,
        help="number of repeats for free adversarial training",
    )
    parser.add_argument(
        "--fgsm-step", default=0.003, type=float, help="step size for FGSM"
    )
    parser.add_argument(
        "--clip-eps",
        default=0.01,
        type=float,
        help="maximum perturbation size (L∞ norm)",
    )
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--save-freq", default=5, type=int, help="save frequency")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="use 20 superclasses instead of 100 classes",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "wrn_32_10"],
        help="model architecture",
    )
    parser.add_argument(
        "--save-dir",
        default="./checkpoints",
        type=str,
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate model on validation set"
    )

    args = parser.parse_args()
    args.clip_eps = args.clip_eps / 255.0
    args.fgsm_step = args.fgsm_step / 255.0

    # Set up data paths
    data_pre_path = args.data_path
    metadata_path = os.path.join(data_pre_path, "meta")
    data_train_path = os.path.join(data_pre_path, "train")
    data_test_path = os.path.join(data_pre_path, "test")

    # Load data
    metadata = unpickle(metadata_path)
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)

    # Get data
    data_train = data_train_dict[b"data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_test = data_test_dict[b"data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    # Get labels based on coarse flag
    print("Using coarse labels" if args.coarse else "Using fine labels")
    if args.coarse:
        LOGGER.info("Using 20 superclasses")
        label_train = np.array(data_train_dict[b"coarse_labels"])
        label_test = np.array(data_test_dict[b"coarse_labels"])
        num_classes = 20
    else:
        LOGGER.info("Using 100 classes")
        label_train = np.array(data_train_dict[b"fine_labels"])
        label_test = np.array(data_test_dict[b"fine_labels"])
        num_classes = 100

    # Set up transformations
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )

    # Create datasets
    train_dataset = CIFAR100Dataset(data_train, label_train, transform=train_transform)
    test_dataset = CIFAR100Dataset(data_test, label_test, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Set up model
    if args.model == "resnet18":
        model = models.resnet18(pretrained=False)
    elif args.model == "resnet34":
        model = models.resnet34(pretrained=False)
    elif args.model == "resnet50":
        model = models.resnet50(pretrained=False)
    elif args.model == "wrn_32_10":
        from Wide_ResNet_32_10 import Model

        model = Model()

    # Adjust the final layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # Define loss function and optimizer
    criterion = (
        nn.CrossEntropyLoss().cuda()
        if torch.cuda.is_available()
        else nn.CrossEntropyLoss()
    )
    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Create adversarial config
    adv_config = {
        "n_repeats": args.n_repeats,
        "fgsm_step": args.fgsm_step,
        "clip_eps": args.clip_eps,
        "print_freq": args.print_freq,
        "save_freq": args.save_freq,
    }

    # Initialize global noise data
    global_noise_data = torch.zeros([args.batch_size, 3, 32, 32])
    if torch.cuda.is_available():
        global_noise_data = global_noise_data.cuda()

    if args.evaluate:
        LOGGER.info("Evaluating model...")
        torch.load(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
        model.eval()
        adv_config["fgsm_step"] = 1 / 255.0
        validate(test_loader, model, criterion)
        validate_pgd(test_loader, model, criterion, adv_config)
        return

    # Main training loop
    args.epochs /= args.n_repeats
    args.epochs = int(args.epochs)
    print(f"Training for {args.epochs} epochs with {args.n_repeats} repeats")
    best_prec1 = 0
    for epoch in range(args.epochs):
        # Adjust learning rate
        adjust_learning_rate(args.lr, optimizer, epoch, args.n_repeats)

        # Train for one epoch
        train_acc = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            adv_config,
            global_noise_data,
        )

        # Evaluate on validation set
        val_acc1, val_acc5 = validate(test_loader, model, criterion)

        # Remember best accuracy and save checkpoint
        is_best = val_acc1 > best_prec1
        best_prec1 = max(val_acc1, best_prec1)

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.model,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                save_dir=args.save_dir,
            )

    LOGGER.info("Training completed!")
    LOGGER.info(f"Best top-1 accuracy: {best_prec1:.2f}%")


if __name__ == "__main__":
    main()
