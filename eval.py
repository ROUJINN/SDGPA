import os
from dataclasses import dataclass, field
from typing import List, Literal

import torch
import torch.nn as nn
import tyro

from models.segmentation.modeling import (
    deeplabv3plus_resnet50,
)
from sgdatasets.acdc import ACDC
from sgdatasets.cityscapes_ext import CityscapesExt
from sgdatasets.gta5 import GTA5DataSet
from utils.helpers import gen_train_dirs, get_test_trans, plot_confusion_matrix
from utils.routines import eval_evaluate, evaluate


@dataclass
class Args:
    cs_path: str
    acdc_path: str
    gta5_path: str
    weight: str
    setting: Literal["day", "fog", "rain", "snow", "night", "game"]

    batch_size: int = 16
    workers: int = 4
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    save_path: str = ""
    save: bool = False

    model: str = "resnet50"
    arch: str = "deeplabv3+"


def main(args):
    print(args)

    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = (512, 1024)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    cs_path = args.cs_path
    acdc_path = args.acdc_path
    gta5_path = args.gta5_path

    testset_day = CityscapesExt(
        cs_path, split="val", target_type="semantic", transforms=test_trans
    )
    testset_snow = ACDC(acdc_path, split="val", transforms=test_trans, ACDC_sub="snow")
    testset_fog = ACDC(acdc_path, split="val", transforms=test_trans, ACDC_sub="fog")
    testset_rain = ACDC(acdc_path, split="val", transforms=test_trans, ACDC_sub="rain")
    testset_night = ACDC(
        acdc_path, split="val", transforms=test_trans, ACDC_sub="night"
    )
    testset_gta5 = GTA5DataSet(
        gta5_path,
        list_path="./sgdatasets/gta5_list/gtav_split_val.txt",
        transforms=test_trans,
    )

    dataloaders = {}
    dataloaders["test_day"] = torch.utils.data.DataLoader(
        testset_day,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_snow"] = torch.utils.data.DataLoader(
        testset_snow,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_fog"] = torch.utils.data.DataLoader(
        testset_fog,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_rain"] = torch.utils.data.DataLoader(
        testset_rain,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_night"] = torch.utils.data.DataLoader(
        testset_night,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )
    dataloaders["test_gta5"] = torch.utils.data.DataLoader(
        testset_gta5,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
    )

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)

    model = deeplabv3plus_resnet50(num_classes=num_classes, backbone_pretrained=False)

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Model pushed to {} GPU(s), type {}.".format(
                torch.cuda.device_count(), torch.cuda.get_device_name(0)
            )
        )

    # Load weights from checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if args.setting == "day":
        print("--- Validation - daytime ---")
        val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(
            dataloaders["test_day"],
            model,
            criterion,
            0,
            CityscapesExt.classLabels,
            CityscapesExt.validClasses,
            void=CityscapesExt.voidClass,
            maskColors=CityscapesExt.maskColors,
            mean=mean,
            std=std,
        )
        print(miou_cs)
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_cs, val_acc_cs)
        plot_confusion_matrix(
            confmat_cs, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("confmat_cs.pdf", bbox_inches="tight")

    elif args.setting == "snow":
        print("--- Validation - ACDC Snow ---")
        test_acc_snow, test_loss_snow, miou_snow, confmat_snow, iousum_snow = (
            eval_evaluate(
                dataloaders["test_snow"],
                model,
                criterion,
                CityscapesExt.classLabels,
                CityscapesExt.validClasses,
                void=CityscapesExt.voidClass,
                maskColors=CityscapesExt.maskColors,
                mean=mean,
                std=std,
                save_root=os.path.join(args.save_path, "snow"),
                save=args.save,
            )
        )
        print(miou_snow)
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_snow, test_acc_snow)
        plot_confusion_matrix(
            confmat_snow, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("confmat_snow.pdf", bbox_inches="tight")

    elif args.setting == "night":
        print("--- Validation - ACDC Night ---")
        test_acc_night, test_loss_night, miou_night, confmat_night, iousum_night = (
            eval_evaluate(
                dataloaders["test_night"],
                model,
                criterion,
                CityscapesExt.classLabels,
                CityscapesExt.validClasses,
                void=CityscapesExt.voidClass,
                maskColors=CityscapesExt.maskColors,
                mean=mean,
                std=std,
                save_root=os.path.join(args.save_path, "night"),
                save=args.save,
            )
        )
        print(miou_night)
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_night, test_acc_night)
        plot_confusion_matrix(
            confmat_night, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("confmat_night.pdf", bbox_inches="tight")

    elif args.setting == "rain":
        print("--- Validation - ACDC Rain ---")
        test_acc_rain, test_loss_rain, miou_rain, confmat_rain, iousum_rain = (
            eval_evaluate(
                dataloaders["test_rain"],
                model,
                criterion,
                CityscapesExt.classLabels,
                CityscapesExt.validClasses,
                void=CityscapesExt.voidClass,
                maskColors=CityscapesExt.maskColors,
                mean=mean,
                std=std,
                save_root=os.path.join(args.save_path, "rain"),
                save=args.save,
            )
        )
        print(miou_rain)
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_rain, test_acc_rain)
        plot_confusion_matrix(
            confmat_rain, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("confmat_rain.pdf", bbox_inches="tight")

    elif args.setting == "fog":
        print("--- Validation - ACDC Fog ---")
        test_acc_fog, test_loss_fog, miou_fog, confmat_fog, iousum_fog = eval_evaluate(
            dataloaders["test_fog"],
            model,
            criterion,
            CityscapesExt.classLabels,
            CityscapesExt.validClasses,
            void=CityscapesExt.voidClass,
            maskColors=CityscapesExt.maskColors,
            mean=mean,
            std=std,
            save_root=os.path.join(args.save_path, "fog"),
            save=args.save,
        )
        print(miou_fog)
        cm_title = "mIoU : {:.3f}, acc : {:.3f}".format(miou_fog, test_acc_fog)
        plot_confusion_matrix(
            confmat_fog, CityscapesExt.classLabels, normalize=True, title=cm_title
        ).savefig("confmat_fog.pdf", bbox_inches="tight")

    elif args.setting == "game":
        print("--- Validation - GTA5 ---")
        test_acc_gta5, test_loss_gta5, miou_gta5, confmat_gta5, iousum_gta5 = (
            eval_evaluate(
                dataloaders["test_gta5"],
                model,
                criterion,
                CityscapesExt.classLabels,
                CityscapesExt.validClasses,
                void=CityscapesExt.voidClass,
                maskColors=CityscapesExt.maskColors,
                mean=mean,
                std=std,
                save_root=os.path.join(args.save_path, "gta5"),
                save=args.save,
            )
        )
        print(miou_gta5)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
