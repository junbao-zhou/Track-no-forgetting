#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import random
import time
import imp
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
import tqdm
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax, lovasz_grad
import tasks.semantic.modules.adf as adf
from tasks.semantic.modules.distill_loss import MSEDistillLoss, KnowledgeDistillationLoss, UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss
from tasks.semantic.task import get_task_labels, get_per_task_classes
from tasks.config import salsanext
from tasks.config.semantic_kitti import learning_ignore
from common.sync_batchnorm.replicate import DataParallelWithCallback


def keep_variance_fn(x):
    return x + 1e-3


def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(
        labels.size(0)), indexes]

    return y_true


def save_to_log(logdir, logfile, message):
    with open(os.path.join(logdir, logfile), "a") as f:
        f.write(message + '\n')
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, os.path.join(logdir, f"SalsaNext{suffix}"))


def convert_model_to_parallel_sync_batchnorm(model: nn.Module):
    model = nn.DataParallel(model)  # spread in gpus
    model = convert_model(model).cuda()  # sync batchnorm
    return model


def dict_to_table_str(dict: dict) -> str:
    str_list = [f"\t{k}: {v}," for k, v in dict.items()]
    return "{\n" + '\n'.join(str_list) + "\n}"


class Trainer():
    def __init__(
            self,
            # ARCH, DATA,
            datadir, logdir, group=0, path=None, uncertainty=False):
        # parameters
        # self.ARCH = ARCH
        # self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.uncertainty = uncertainty

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            batch_size = salsanext.train.batch_size_per_GPU * \
                torch.cuda.device_count()
            self.print_save_to_log(f"gpu number = {torch.cuda.device_count()}")
            self.print_save_to_log(f"batch_size = {batch_size}")

        from tasks.semantic.dataset.kitti import parser
        # get the data
        self.parser = parser.Parser(
            root=self.datadir,
            # datargs=self.DATA,
            # archargs=self.ARCH,
            batch_size=batch_size,
            is_test=False,
            gt=True,
            shuffle_train=True,
        )
        self.print_save_to_log(
            f"Parser's learning map = \n{dict_to_table_str(self.parser.learning_map)}")
        self.print_save_to_log(
            f'label_frequencies = \n{self.parser.xentropy_label_frequencies}')

        self.ignore_classes = [
            class_i for class_i, is_ignored in learning_ignore.items() if is_ignored]
        self.print_save_to_log(f"Ignored Classes = {self.ignore_classes}")
        assert len(self.ignore_classes) <= 1

        # weights for loss (and bias)
        epsilon_w = salsanext.train.loss.epsilon_w
        self.loss_weight = 1 / \
            (self.parser.xentropy_label_frequencies + epsilon_w)  # get weights
        self.print_save_to_log(
            f"Loss weights from label_frequencies = \n{self.loss_weight.data}")
        # exit(0)

        per_task_classes = get_per_task_classes(
            salsanext.train.task_name, salsanext.train.task_step)

        self.model_old = None
        with torch.no_grad():
            self.model = IncrementalSalsaNext(per_task_classes)
            self.print_save_to_log(f'Initialize model')
            step = salsanext.train.task_step
            if step > 0 and salsanext.train.is_use_base_model:
                self.model_old = IncrementalSalsaNext(
                    [per_task_classes[step-1]])
                self.print_save_to_log(f'Initialize old model')

        # base_model_path = salsanext.train.base_model
        if os.path.exists(salsanext.train.base_model):
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(
                os.path.join(salsanext.train.base_model,
                             "SalsaNext_valid_best"),
                map_location=lambda storage, loc: storage,
            )
            if self.model_old is not None:
                self.print_save_to_log(f'load state dict to old model')
                self.model_old.load_state_dict(
                    w_dict['state_dict'], strict=True)

        if os.path.exists(salsanext.train.novel_model):
            w_dict = torch.load(
                os.path.join(
                    salsanext.train.novel_model, "SalsaNext_valid_best"),
                map_location=lambda storage, loc: storage,
            )
            self.print_save_to_log(f'load state dict to model')
            self.model.load_state_dict(w_dict['state_dict'], strict=False)

        self.tb_logger = Logger(os.path.join(self.log, "tb"))

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("cuda is available")
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
            if self.model_old is not None:
                self.model_old.cuda()
        # Multi Gpus?
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = convert_model_to_parallel_sync_batchnorm(self.model)
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()
            if self.model_old is not None:
                self.model_old = convert_model_to_parallel_sync_batchnorm(
                    self.model_old)

        self.criterion = UnbiasedCrossEntropy(old_cl=per_task_classes[0], weight=self.loss_weight).to(self.device)\
            if salsanext.train.task_step > 0 else \
            nn.NLLLoss(
                weight=self.loss_weight,
                ignore_index=-100 if len(self.ignore_classes) == 0 else self.ignore_classes[0]).to(self.device)
        self.print_save_to_log(
            f'self.criterion : {self.criterion}, ignore_index : {self.criterion.ignore_index}')
        self.ls = Lovasz_softmax(
            ignore=None if len(self.ignore_classes) == 0 else self.ignore_classes[0]).to(self.device)
        self.print_save_to_log(
            f'self.lovasz.ignore : {self.ls.ignore}')
        if self.model_old is not None:
            self.print_save_to_log(
                f'Distill Loss : {salsanext.train.loss.distill_name.__name__}')
            self.distill = salsanext.train.loss.distill_name().to(self.device)

        self.loss_coefficient = salsanext.train.loss.coefficient

        # loss as dataparallel too (more images in batch)
        # if self.n_gpus > 1:
        #     self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
        #     self.ls = nn.DataParallel(self.ls).cuda()
        #     self.distill = nn.DataParallel(self.distill).cuda()
        self.optimizer = optim.SGD(
            [{'params': self.model.parameters()}],
            lr=salsanext.train.optimizer.lr,
            momentum=salsanext.train.optimizer.momentum,
            weight_decay=salsanext.train.optimizer.w_decay,
        )

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(salsanext.train.optimizer.wup_epochs * steps_per_epoch)
        final_decay = salsanext.train.optimizer.lr_decay ** (
            1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=salsanext.train.optimizer.lr,
                                  warmup_steps=up_steps,
                                  momentum=salsanext.train.optimizer.momentum,
                                  decay=final_decay)

        # put the old model into distributed memory and freeze it
        if self.model_old is not None:
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

    def calculate_estimate(self, epoch, iter):
        estimate = int(
            (self.data_time_t.avg + self.batch_time_t.avg) *
            (self.parser.get_train_size() * salsanext.train.max_epochs -
             (iter + 1 + epoch * self.parser.get_train_size()))) + \
            int(
                self.batch_time_e.avg * self.parser.get_valid_size() * (
                    salsanext.train.max_epochs - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def make_log_img_torch(proj_mask, in_vol, pred_argmax, proj_labels, color_fn):
        mask_np = proj_mask.cpu().numpy()
        depth_np = in_vol[0].cpu().numpy()
        pred_np = pred_argmax.cpu().numpy()
        gt_np = proj_labels.cpu().numpy()
        out = Trainer.make_log_img(
            depth_np, mask_np, pred_np, gt_np, color_fn)
        return out

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def print_save_to_log(self, message):
        print(message)
        save_to_log(self.log, 'print.log', message)

    def train(self):

        self.print_save_to_log(f"Ignoring class {self.ignore_classes} in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_classes)

        # train for n epochs
        for epoch in range(self.epoch, salsanext.train.max_epochs):
            # train for 1 epoch
            acc, iou, loss, update_mean, hetero_l = self.train_epoch(
                train_loader=self.parser.get_train_set(),
                model=self.model,
                model_old=self.model_old,
                criterion=self.criterion,
                optimizer=self.optimizer,
                epoch=epoch,
                evaluator=self.evaluator,
                scheduler=self.scheduler,
                color_fn=self.parser.to_color,
                report=salsanext.train.report_batch,
                show_scans=salsanext.train.show_scans,
            )

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["train_hetero"] = hetero_l

            # remember best iou and save checkpoint
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if type(self.model) == DataParallelWithCallback else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'info': self.info,
                'scheduler': self.scheduler.state_dict(),
            }
            save_checkpoint(state, self.log, suffix="")

            if self.info['train_iou'] > self.info['best_train_iou']:
                self.print_save_to_log(
                    "Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state['info'] = self.info
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % salsanext.train.report_epoch == 0:
                # evaluate on validation set
                self.print_save_to_log("*" * 80)
                acc, iou, loss, rand_img, hetero_l = self.validate(
                    val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    criterion=self.criterion,
                    evaluator=self.evaluator,
                    class_func=self.parser.get_xentropy_class_string,
                    color_fn=self.parser.to_color,
                    save_scans=salsanext.train.save_scans,
                )

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou
                self.info['valid_heteros'] = hetero_l

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                self.print_save_to_log(
                    "Best mean iou in validation so far, save model!")
                self.print_save_to_log("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state['info'] = self.info
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            Trainer.save_to_log(
                logdir=self.log,
                logger=self.tb_logger,
                info=self.info,
                epoch=epoch,
                w_summary=salsanext.train.save_summary,
                model=self.model_single,
                img_summary=salsanext.train.save_scans,
                imgs=rand_img,
            )

        self.print_save_to_log('Finished Training')

        return

    def train_epoch(self, train_loader, model, model_old, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        print(f"Epoch begin")
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        # switch to train mode
        model.train()

        end = time.time()

        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                #proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

            if model_old is not None:
                with torch.no_grad():
                    output_old, logits_old, decode_result_old = model_old(
                        in_vol)

            # compute output
            output, logits, decode_result = model(in_vol)

            if type(criterion) == nn.NLLLoss:
                nll_loss = self.loss_coefficient.NLLLoss * criterion(
                    torch.log(output.clamp(min=1e-8)), proj_labels)
            else:
                nll_loss = self.loss_coefficient.NLLLoss * criterion(
                    logits, proj_labels)
            lovasz_loss = self.loss_coefficient.Lovasz_softmax * \
                self.ls(output, proj_labels.long())
            loss_m = nll_loss + lovasz_loss

            if model_old is not None:
                if type(self.distill) == MSEDistillLoss:
                    distill_loss = self.distill(
                        decode_result, decode_result_old)
                else:
                    # print(f"self.distill is {type(self.distill)}")
                    distill_loss = self.distill(logits, logits_old)
                loss_m += self.loss_coefficient.Distill * distill_loss

            # print(f'loss_m.size() = {loss_m.size()}')
            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                # loss_m.backward(idx)
                loss_m.backward()
            else:
                loss_m.backward()
            optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(
                            value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                out = Trainer.make_log_img_torch(
                    proj_mask[0],
                    in_vol[0],
                    argmax[0],
                    proj_labels[0],
                    color_fn,
                )
                out2 = Trainer.make_log_img_torch(
                    proj_mask[1],
                    in_vol[1],
                    argmax[1],
                    proj_labels[1],
                    color_fn,
                )

                out = np.concatenate([out, out2], axis=0)
                cv2.imshow("sample_training", out)
                cv2.waitKey(1)

            if i % salsanext.train.report_batch == 0:
                self.print_save_to_log(f"""\
Lr: {lr:.3e} | \
Update: {update_mean:.3e} mean, {update_std:.3e} std | \
Epoch: [{epoch}][{i}/{len(train_loader)}] | \
Time {self.batch_time_t.val:.3f} ({self.batch_time_t.avg:.3f}) | \
Data {self.data_time_t.val:.3f} ({self.data_time_t.avg:.3f}) | \
Loss {losses.val:.4f} ({losses.avg:.4f}) | \
acc {acc.val:.3f} ({acc.avg:.3f}) | \
IoU {iou.val:.3f} ({iou.avg:.3f}) | [{self.calculate_estimate(epoch, i)}]""")

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        print(f"validate begin")
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(tqdm.tqdm(val_loader)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                output, logits, decode_result = model(in_vol)
                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))

                wces.update(wce.mean().item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    out = Trainer.make_log_img_torch(
                        proj_mask[0],
                        in_vol[0],
                        argmax[0],
                        proj_labels[0],
                        color_fn,
                    )
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            log_txt = f"""
Validation set:
Time avg per batch {self.batch_time_e.avg:.3f}
Loss avg {losses.avg:.4f}
Jaccard avg {jaccs.avg:.4f}
WCE avg {wces.avg:.4f}
Acc avg {acc.avg:.3f}
IoU avg {iou.avg:.3f}
"""
            if self.uncertainty:
                log_txt += f"""\
Hetero avg {hetero_l.avg:.4f}
"""
            self.print_save_to_log(log_txt)
            # print also classwise

            for i, jacc in enumerate(class_jaccard):
                self.print_save_to_log(
                    f'IoU class {i:} [{class_func(i):}] = {jacc:.3f}')
                self.info["valid_classes/" + class_func(i)] = jacc

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg
