# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:38:31 2019 by Attila Lengyel - attila@lengyel.nl
"""

import os
import random
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms

from utils.get_iou import iouCalc
from utils.helpers import AverageMeter, ProgressMeter, visim, vislbl


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train_epoch(dataloader, model, criterion, optimizer, epoch, log, void=-1):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_running = AverageMeter("Loss", ":.4e")
    acc_running = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Epoch: [{}]".format(epoch),
    )

    # set model in training mode
    model.train()

    end = time.time()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, inputs_night, labels) in enumerate(dataloader):
            data_time.update(time.time() - end)

            # input resolution
            res = inputs.shape[2] * inputs.shape[3]

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            # Statistics
            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels == void).sum())
            acc = corrects.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)

            # Output training info
            progress.display(epoch_step)
            # Append current stats to csv
            with open("logs/log_batch.csv", "a") as log_batch:
                log_batch.write(
                    "{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
                        epoch,
                        epoch_step,
                        loss / bs,
                        loss_running.avg,
                        acc,
                        acc_running.avg,
                    )
                )

            batch_time.update(time.time() - end)
            end = time.time()

    log.info(
        "Epoch {} train loss: {:.4f}, acc: {:.4f}".format(
            epoch, loss_running.avg, acc_running.avg
        )
    )

    return loss_running.avg, acc_running.avg


def train_epoch_similarity_dual(
    dataloader, model, criterion, optimizer, epoch, log, void=-1, sim_weight=0.1
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_running = AverageMeter("Loss", ":.4e")
    loss_source_running = AverageMeter("Loss_source", ":.4e")
    loss_synth_running = AverageMeter("Loss_synth", ":.4e")
    loss_similarity_running = AverageMeter("Loss_similarity", ":.4e")
    acc_running = AverageMeter("Acc", ":6.2f")
    acc_synth_running = AverageMeter("Acc_synth", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [
            batch_time,
            data_time,
            loss_running,
            loss_source_running,
            loss_synth_running,
            loss_similarity_running,
            acc_running,
            acc_synth_running,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # set model in training mode
    model.train()

    end = time.time()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, inputs_synth, labels) in enumerate(dataloader):
            bs = inputs.size(0)  # current batch size
            data_time.update(time.time() - end)

            # input resolution
            res = inputs.shape[2] * inputs.shape[3]
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            optimizer.zero_grad()
            # forward & backward day data (pass day data to normal route)
            loss, loss_similarity = 0.0, 0.0
            outputs, outputs_synth, feats_q, feats_z = model(
                inputs, inputs_synth
            )  # encoder_k forwards night data
            for i in range(len(feats_q)):
                loss_similarity += (
                    -1 * (feats_q[i] * feats_z[i]).sum(dim=1).mean() * sim_weight
                )

            loss_similarity /= len(feats_q)
            loss += loss_similarity

            preds = torch.argmax(outputs, 1)
            loss_source = criterion(outputs, labels)
            loss += loss_source

            preds_synth = torch.argmax(outputs_synth, 1)
            loss_synth = criterion(outputs_synth, labels)
            loss += loss_synth

            loss.backward()

            loss = loss.item()
            loss_running.update(loss, bs)
            loss_source_running.update(loss_source.item(), bs)
            loss_similarity_running.update(loss_similarity.item(), bs)
            loss_synth_running.update(loss_synth.item(), bs)

            # update parameters
            optimizer.step()

            # Statistics
            nvoid = int((labels == void).sum())
            corrects = torch.sum(preds == labels.data)
            acc = corrects.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            corrects_synth = torch.sum(preds_synth == labels.data)
            acc_synth = corrects_synth.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_synth_running.update(acc_synth, bs)

            # Output training info
            progress.display(epoch_step)
            # Append current stats to csv
            with open("logs/log_batch.csv", "a") as log_batch:
                log_batch.write(
                    "{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
                        epoch,
                        epoch_step,
                        loss / bs,
                        loss_running.avg,
                        loss_source_running.avg,
                        loss_synth_running.avg,
                        loss_similarity_running.avg,
                        acc_synth_running.avg,
                        acc_running.avg,
                    )
                )

            batch_time.update(time.time() - end)
            end = time.time()

    log.info(
        "Epoch {} train loss: {:.4f}, acc: {:.4f}".format(
            epoch, loss_running.avg, acc_running.avg
        )
    )

    return (
        loss_running.avg,
        loss_source_running.avg,
        loss_synth_running.avg,
        loss_similarity_running.avg,
        acc_running.avg,
        acc_synth_running.avg,
    )


def evaluate(
    dataloader,
    model,
    criterion,
    epoch,
    classLabels,
    validClasses,
    log=None,
    void=-1,
    maskColors=None,
    mean=None,
    std=None,
):
    iou = iouCalc(classLabels, validClasses, voidClass=void)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_running = AverageMeter("Loss", ":.4e")
    acc_running = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader), [batch_time, loss_running, acc_running], prefix="Test: "
    )

    # set model in training mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time() - end)

            # input resolution
            res = inputs.shape[2] * inputs.shape[3]

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels == void).sum())
            acc = corrects.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)

            # # Save visualizations of first batch
            # if epoch_step == 0 and maskColors is not None:
            # 	for i in range(inputs.size(0)):
            # 		filename = os.path.splitext(os.path.basename(filepath[i]))[0]
            # 		# Only save inputs and labels once
            # 		if epoch == 0:
            # 			img = visim(inputs[i,:,:,:], mean, std)
            # 			label = vislbl(labels[i,:,:], maskColors)
            # 			if len(img.shape) == 3:
            # 				cv2.imwrite('images/{}.png'.format(filename),img[:,:,::-1])
            # 			else:
            # 				cv2.imwrite('images/{}.png'.format(filename),img)
            # 			cv2.imwrite('images/{}_gt.png'.format(filename),label[:,:,::-1])
            # 		# Save predictions
            # 		pred = vislbl(preds[i,:,:], maskColors)
            # 		cv2.imwrite('images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress info
            progress.display(epoch_step)

        miou, iou_summary, confMatrix = iou.outputScores(epoch=epoch)
        if log is not None:
            log.info(" * Acc {:.3f}".format(acc_running.avg))
        print(iou_summary)

    return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary


def eval_evaluate(
    dataloader,
    model,
    criterion,
    classLabels,
    validClasses,
    log=None,
    void=-1,
    maskColors=None,
    mean=None,
    std=None,
    save_root="",
    save_suffix="",
    save=False,
):
    save_path = os.path.join("results", save_root)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # paths = [
    #     os.path.join(save_path, "gt"),
    #     os.path.join(save_path, "visuals"),
    #     os.path.join(save_path, "original"),
    #     # os.path.join(save_path,'labelTrainIds_invalid'),os.path.join(save_path,'confidence'),
    #     # os.path.join(save_path,'labelTrainIds')
    # ]
    # for p in paths:
    #     if not os.path.exists(p):
    #         os.makedirs(p)

    iou = iouCalc(classLabels, validClasses, voidClass=void)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_running = AverageMeter("Loss", ":.4e")
    acc_running = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader), [batch_time, loss_running, acc_running], prefix="Test: "
    )

    # set model in training mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time() - end)

            # input resolution
            h, w = labels.shape[-2:]
            res = h * w

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            confidence = torch.softmax(outputs, 1).max(1)[0]
            loss = criterion(outputs, labels)
            # preds = transforms.Resize((h,w), interpolation=Image.NEAREST)(preds)
            # loss = criterion(outputs, transforms.Resize((512,1024), interpolation=Image.NEAREST)(labels))

            # Statistics
            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels == void).sum())
            acc = corrects.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)

            # Save visualizations
            if save:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]

                    img = visim(inputs[i, :, :, :], mean, std)
                    label = vislbl(labels[i, :, :], maskColors)
                    pred = vislbl(preds[i, :, :], maskColors)

                    cv2.imwrite(
                        save_path + f"/visuals/{filename}-{save_suffix}-{acc:.2f}.png",
                        pred[:, :, ::-1],
                    )
                    cv2.imwrite(
                        save_path + f"/visuals/{filename}-original.png", img[:, :, ::-1]
                    )
                    cv2.imwrite(
                        save_path + f"/visuals/{filename}-gt.png", label[:, :, ::-1]
                    )
                    # Save predictions
                    # preds = transforms.Resize(input_shape, interpolation=Image.NEAREST)(preds)
                    pred = vislbl(preds[i, :, :], maskColors)
                    cv2.imwrite(
                        save_path + "/labelTrainIds/{}.png".format(filename),
                        pred[:, :, ::-1],
                    )
                    cv2.imwrite(
                        save_path + "/labelTrainIds_invalid/{}.png".format(filename),
                        pred[:, :, ::-1],
                    )
                    con = (confidence[i].cpu().numpy() * 65536).astype(np.uint16)
                    cv2.imwrite(save_path + "/confidence/{}.png".format(filename), con)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress info
            progress.display(epoch_step)

        miou, iou_summary, confMatrix = iou.outputScores(epoch=0)
        if log is not None:
            log.info(" * Acc {:.3f}".format(acc_running.avg))
        print(iou_summary)

    return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary


def test_evaluate(
    dataloader,
    model,
    criterion,
    classLabels,
    validClasses,
    log=None,
    void=-1,
    maskColors=None,
    mean=None,
    std=None,
    save_root="",
    input_shape=(1080, 1920),
):
    save_path = os.path.join("results", save_root)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    paths = [
        os.path.join(save_path, "labelTrainIds"),
        # os.path.join(save_path,'img'),
        os.path.join(save_path, "labelTrainIds_invalid"),
        os.path.join(save_path, "confidence"),
        os.path.join(save_path, "visuals"),
    ]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(len(dataloader), [batch_time], prefix="Test: ")

    # set model in training mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, filepath) in enumerate(dataloader):
            data_time.update(time.time() - end)

            inputs = inputs.float().cuda()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            confidence = torch.softmax(outputs, 1).max(1)[0]
            # loss = criterion(outputs, transforms.Resize((512,1024), interpolation=Image.NEAREST)(labels))

            # Statistics
            bs = inputs.size(0)  # current batch size

            # Save visualizations
            for i in range(inputs.size(0)):
                filename = os.path.splitext(os.path.basename(filepath[i]))[0]

                # img = visim(inputs[i,:,:,:], mean, std)
                # if len(img.shape) == 3:
                # 	cv2.imwrite(save_path + '/visuals/{}.png'.format(filename),img[:,:,::-1])
                # else:
                # 	cv2.imwrite(save_path + '/visuals/{}.png'.format(filename),img)
                # Save predictions
                preds_i = transforms.Resize(input_shape, interpolation=Image.NEAREST)(
                    preds[i : i + 1]
                )  # resize to original size (1080,1920)
                preds_i = preds_i.permute(1, 2, 0).cpu().numpy()

                cv2.imwrite(
                    save_path + "/labelTrainIds/{}.png".format(filename), preds_i
                )
                cv2.imwrite(
                    save_path + "/labelTrainIds_invalid/{}.png".format(filename),
                    preds_i,
                )  # dummy
                cv2.imwrite(
                    save_path + "/confidence/{}.png".format(filename), preds_i
                )  # dummy

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress info
            progress.display(epoch_step)

        os.chdir(save_path)
        os.system("zip -r results.zip labelTrainIds labelTrainIds_invalid confidence")


def evaluate_single(
    dataloader,
    model,
    criterion,
    epoch,
    classLabels,
    validClasses,
    log=None,
    void=-1,
    maskColors=None,
    mean=None,
    std=None,
):
    iou = iouCalc(classLabels, validClasses, voidClass=void)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_running = AverageMeter("Loss", ":.4e")
    acc_running = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader), [batch_time, loss_running, acc_running], prefix="Test: "
    )

    # set model in training mode
    model.eval()
    mious = []
    indexs = []
    filepaths = []

    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time() - end)

            # input resolution
            res = inputs.shape[2] * inputs.shape[3]

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels == void).sum())
            acc = corrects.double() / (
                bs * res - nvoid
            )  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)

            miou, iou_summary, confMatrix = iou.outputScores(epoch=epoch)
            mious.append(miou)
            indexs.append(epoch_step)
            filepaths.append(filepath)
            iou.clear()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress info
            progress.display(epoch_step)

    print(len(mious))
    np.save("mious.npy", mious)
    np.save("indexs.npy", indexs)
    np.save("filepaths.npy", filepaths)
    return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary
