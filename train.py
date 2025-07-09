import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tyro
from torch.utils.data import Subset

from args import Args
from models.segmentation.modeling import deeplabv3plus_similarity_resnet50
from sgdatasets.acdc import ACDC
from sgdatasets.cityscapes_ext import CityscapesExt
from sgdatasets.gta5 import GTA5DataSet
from sgdatasets.paired_city import CityTwoDomains
from utils.helpers import (
    gen_train_dirs,
    get_multi_train_trans,
    get_test_trans,
    plot_confusion_matrix,
)
from utils.routines import (
    evaluate,
    train_epoch_similarity_dual,
)
from utils.scheduler import PolyLR


class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), "a") as f:
            f.write(msg + "\n")


def main(args: Args):
    # Configure dataset paths here, paths should look like this
    source_path = args.source_path
    synth_path = args.synth_path
    target_path = args.target_path

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # Generate training directories
    gen_train_dirs(args.experiment_name)
    args.save()

    log = logger("./")
    # Generate log files
    with open("logs/log_batch.csv", "a") as batch_log:
        batch_log.write(
            "epoch, epoch step, train loss, avg train loss, avg train source loss, avg train sim loss, train acc, avg train acc\n"
        )
    with open("logs/log_epoch.csv", "a") as epoch_log:
        epoch_log.write(
            "epoch, train loss, train source loss, train synth loss, train sim loss, val loss source, train acc, train acc synth, val acc source, test target acc, miou, miou target, learning rate, scale \n"
        )

    # Initialize metrics
    best_miou = 0.0
    best_miou_target = 0.0
    metrics = {
        "train_loss": [],
        "train_loss_source": [],
        "train_loss_synth": [],
        "train_loss_sim": [],
        "train_acc": [],
        "train_synth_acc": [],
        "val_acc_cs": [],
        "val_loss_cs": [],
        "miou_cs": [],
        "test_acc_target": [],
        "test_loss_target": [],
        "miou_target": [],
    }
    start_epoch = 0

    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = args.target_size
    crop_size = args.crop_size

    train_trans = get_multi_train_trans(
        mean,
        std,
        target_size,
        crop_size,
        args.jitter,
        args.scale,
        True,
        args.blur,
        args.cutout,
    )

    test_trans = get_test_trans(mean, std, target_size)

    trainset = CityTwoDomains(
        root=source_path, root_new_domain=synth_path, transforms=train_trans
    )
    valset = CityscapesExt(
        source_path, split="val", target_type="semantic", transforms=test_trans
    )

    if args.target_type == "acdc_night":
        testset_target = ACDC(
            target_path, split="val", transforms=test_trans, ACDC_sub="night"
        )
    elif args.target_type == "acdc_snow":
        testset_target = ACDC(
            target_path, split="val", transforms=test_trans, ACDC_sub="snow"
        )
    elif args.target_type == "acdc_rain":
        testset_target = ACDC(
            target_path, split="val", transforms=test_trans, ACDC_sub="rain"
        )
    elif args.target_type == "acdc_fog":
        testset_target = ACDC(
            target_path, split="val", transforms=test_trans, ACDC_sub="fog"
        )
    elif args.target_type == "gta5":
        list_path = "/home/zhaozijing/luojun/model_adaptation/sgdatasets/gta5_list/gtav_split_val.txt"
        testset_target = GTA5DataSet(
            target_path, list_path=list_path, transforms=test_trans
        )
    else:
        raise ValueError

    if args.debug:
        trainset = Subset(trainset, list(range(5)))
        valset = Subset(valset, list(range(5)))
        testset_target = Subset(testset_target, list(range(5)))

    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=True,
    )
    dataloaders["val"] = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_target"] = torch.utils.data.DataLoader(
        testset_target,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)

    if args.load is None:
        model = deeplabv3plus_similarity_resnet50(
            num_classes=num_classes, backbone_pretrained=True
        )
    else:
        model = deeplabv3plus_similarity_resnet50(
            num_classes=num_classes, backbone_pretrained=False
        )

    if torch.cuda.is_available():  # Push model to GPU
        model = torch.nn.DataParallel(model).cuda()
        log.info(
            "Model pushed to {} GPU(s), type {}.".format(
                torch.cuda.device_count(), torch.cuda.get_device_name(0)
            )
        )

    if args.load is not None:
        state_dict = torch.load(args.load)["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        log.info("Loaded pretrained model from {}".format(args.load))

    params_1 = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("head" in name or "pred" in name):
            params_1.append(param)

    params_2 = []
    for name, param in model.named_parameters():
        if param.requires_grad and not ("head" in name or "pred" in name):
            params_2.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": params_1, "lr": args.lr_head},
            {"params": params_2, "lr": args.lr},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    # Prepare schedular
    if args.schedular == "poly":
        scheduler = PolyLR(optimizer, args.epochs, power=0.9)

    # Resume training from checkpoint
    if args.resume:
        log.info("Resuming training from {}.".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        log.info(f"Current LR: {optimizer.param_groups[0]['lr']}")
        start_epoch = checkpoint["epoch"] + 1
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_miou_target = checkpoint["best_miou_target"]
            best_miou = checkpoint["best_miou"]
            for _ in range(start_epoch):
                scheduler.step()
        except:
            log.info(
                "Could not load optimizer state dict. Initializing optimizer from scratch."
            )
            epoch = checkpoint["epoch"]
            best_miou_target = 0
            best_miou = 0
            for _ in range(
                start_epoch
            ):  # 训练第start_epoch之前，需要step start_epoch次
                scheduler.step()

    since = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Train
        log.info("--- Training ---")
        (
            train_loss,
            train_loss_source,
            train_loss_synth,
            train_loss_sim,
            train_acc,
            train_synth_acc,
        ) = train_epoch_similarity_dual(
            dataloaders["train"],
            model,
            criterion,
            optimizer,
            epoch,
            log,
            CityscapesExt.voidClass,
            args.sim_weight,
        )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        metrics["train_loss"].append(train_loss)
        metrics["train_loss_source"].append(train_loss_source)
        metrics["train_loss_synth"].append(train_loss_synth)
        metrics["train_loss_sim"].append(train_loss_sim)
        metrics["train_acc"].append(train_acc)
        metrics["train_synth_acc"].append(train_synth_acc)

        # Validate
        log.info("--- Validation - Source ---")
        val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(
            dataloaders["val"],
            model,
            criterion,
            epoch,
            CityscapesExt.classLabels,
            CityscapesExt.validClasses,
            log,
            void=CityscapesExt.voidClass,
            maskColors=CityscapesExt.maskColors,
            mean=mean,
            std=std,
        )
        log.info("--- Validation - Target ---")
        (
            test_acc_target,
            test_loss_target,
            miou_target,
            confmat_target,
            iousum_target,
        ) = evaluate(
            dataloaders["test_target"],
            model,
            criterion,
            epoch,
            CityscapesExt.classLabels,
            CityscapesExt.validClasses,
            log,
            void=CityscapesExt.voidClass,
            maskColors=CityscapesExt.maskColors,
            mean=mean,
            std=std,
        )
        metrics["val_acc_cs"].append(val_acc_cs)
        metrics["val_loss_cs"].append(val_loss_cs)
        metrics["miou_cs"].append(miou_cs)
        metrics["test_acc_target"].append(test_acc_target)
        metrics["test_loss_target"].append(test_loss_target)
        metrics["miou_target"].append(miou_target)

        # Write logs
        with open("logs/log_epoch.csv", "a") as epoch_log:
            epoch_log.write(
                "{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                    epoch,
                    train_loss,
                    train_loss_source,
                    train_loss_synth,
                    train_loss_sim,
                    val_loss_cs,
                    train_acc,
                    train_synth_acc,
                    val_acc_cs,
                    test_acc_target,
                    miou_cs,
                    miou_target,
                    lr,
                )
            )
        with open("logs/class_iou.txt", "w") as ioufile:
            ioufile.write(iousum_cs)
            ioufile.write(iousum_target)
        # Plot confusion matrices
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_cs, val_acc_cs)
        plot_confusion_matrix(
            confmat_cs, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("logs/confmat_cs.pdf", bbox_inches="tight")
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_target, test_acc_target)
        plot_confusion_matrix(
            confmat_target, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("logs/confmat_target.pdf", bbox_inches="tight")

        if epoch == args.stop_epoch - 1:
            shutil.copy(
                "logs/confmat_cs.pdf", f"logs/target_confmat_cs_{args.stop_epoch}.pdf"
            )  # save confmat
            shutil.copy(
                "logs/confmat_target.pdf", f"logs/target_confmat_target_{args.stop_epoch}.pdf"
            )  # save confmat
            shutil.copy("logs/class_iou.txt", f"logs/target_class_iou_{args.stop_epoch}.txt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                f"weights/weights_{args.stop_epoch}.pth.tar",
            )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou_target": best_miou_target,
                "best_miou": best_miou,
                "metrics": metrics,
            },
            "weights/checkpoint.pth.tar",
        )

        if miou_target > best_miou_target:
            log.info(
                "mIoU improved from {:.4f} to {:.4f}.".format(
                    best_miou_target, miou_target
                )
            )
            best_miou_target = miou_target
            best_acc_target = test_acc_target  # acc corresponding to the best miou
            shutil.copy(
                "logs/confmat_cs.pdf", "logs/best_target_confmat_cs.pdf"
            )  # save confmat
            shutil.copy(
                "logs/confmat_target.pdf", "logs/best_target_confmat_target.pdf"
            )  # save confmat
            shutil.copy("logs/class_iou.txt", "logs/best_target_class_iou.txt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                "weights/best_weights_target.pth.tar",
            )

        if miou_cs > best_miou:
            log.info("mIoU improved from {:.4f} to {:.4f}.".format(best_miou, miou_cs))
            best_miou = miou_cs
            best_acc = val_acc_cs  # acc corresponding to the best miou
            shutil.copy(
                "logs/confmat_cs.pdf", "logs/best_confmat_cs.pdf"
            )  # save confmat
            shutil.copy(
                "logs/confmat_target.pdf", "logs/best_confmat_target.pdf"
            )  # save confmat
            shutil.copy("logs/class_iou.txt", "logs/best_class_iou.txt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                "weights/best_weights.pth.tar",
            )

    time_elapsed = time.time() - since
    log.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    log.info("Best mIoU target: {:4f}".format(best_miou_target))

    # Plot learning curves
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("miou")
    ln1 = ax1.plot(x, metrics["miou_cs"], color="tab:red")
    ln2 = ax1.plot(x, metrics["miou_target"], color="tab:green")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel("accuracy")
    o = []
    for tensor in metrics["val_acc_cs"]:
        o.append(tensor.cpu().item())
    ln4 = ax2.plot(x, o, color="tab:red", linestyle="dashed")
    ax2.set_ylabel("accuracy")
    o2 = []
    for tensor in metrics["test_acc_target"]:
        o2.append(tensor.cpu().item())
    ln5 = ax2.plot(x, o2, color="tab:green", linestyle="dashed")
    lns = ln1 + ln2 + ln4 + ln5
    plt.legend(lns, ["CS mIoU", "Target mIoU", "CS Accuracy", "Target Accuracy"])
    plt.tight_layout()
    plt.savefig("logs/learning_curve.pdf")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
