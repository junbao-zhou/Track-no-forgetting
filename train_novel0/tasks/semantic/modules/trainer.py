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
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax, lovasz_grad
import tasks.semantic.modules.adf as adf
from tasks.semantic.modules.distill_loss import UnbiasedKnowledgeDistillationLoss
from tasks.semantic.task import get_task_labels, get_per_task_classes

def keep_variance_fn(x):
    return x + 1e-3

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNext" + suffix)


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, group=0, path=None, uncertainty=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
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

        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')

        self.parser = parserModule.Parser(root=self.datadir,
                                          datargs = self.DATA,
                                          archargs = self.ARCH,
                                          gt=True,
                                          shuffle_train=True)

        # weights for loss (and bias)
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        print(f'label_frequencies = {self.parser.xentropy_label_frequencies}')
        # exit()
        self.loss_w = 1 / (self.parser.xentropy_label_frequencies + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from label_frequencies: ", self.loss_w.data)
        exit(0)

        self.model_old = None
        with torch.no_grad():
            nclasses = get_per_task_classes(self.ARCH["train"]["task_name"], self.ARCH["train"]['task_step'])
            self.model = IncrementalSalsaNext(nclasses)
            # exit(0)
            step = self.ARCH["train"]["task_step"] 
            if step > 0:
                self.model_old = IncrementalSalsaNext([nclasses[step-1]])

        self.tb_logger = Logger(self.log + "/tb")

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
            if self.model_old is not None:
                self.model_old.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()
            if self.model_old is not None:
                self.model_old = nn.DataParallel(self.model_old)  # spread in gpus
                self.model_old = convert_model(self.model_old).cuda()  # sync batchnorm  
        
        # if self.model_old is not None:
        #     for layer in self.model_old.logits:
        #         layer.to(self.device)
        # for layer in self.model.logits:
        #     layer.to(self.device)        

        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        self.distill = UnbiasedKnowledgeDistillationLoss().to(self.device)

        self.loss_coefficient = self.ARCH["train"]["loss_coefficient"]

        # loss as dataparallel too (more images in batch)
        # if self.n_gpus > 1:
        #     self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
        #     self.ls = nn.DataParallel(self.ls).cuda()
        #     self.distill = nn.DataParallel(self.distill).cuda()
        self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

        base_model_path = self.ARCH["train"]["base_model"]
        if os.path.exists(base_model_path):
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(base_model_path + "/SalsaNext_valid_best",
                                map_location=lambda storage, loc: storage)
            # print(w_dict['state_dict'].keys())
            # exit(0)
            if self.model_old is not None:
                print(f'load state dict to old model')
                self.model_old.load_state_dict(w_dict['state_dict'], strict=True)

            print(f'load state dict to model')
            self.model.load_state_dict(w_dict['state_dict'], strict=False)
       
        # put the old model into distributed memory and freeze it
        if self.model_old is not None:
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
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

    def train(self):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # train for 1 epoch
            acc, iou, loss, update_mean,hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           model_old=self.model_old,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])

            # update info
            self.info["train_update"] = update_mean
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["train_hetero"] = hetero_l

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")

            if self.info['train_iou'] > self.info['best_train_iou']:
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img,hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou
                self.info['valid_heteros'] = hetero_l

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger,
                                info=self.info,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=rand_img)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, model_old, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
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
                    output_old, logits_old = model_old(in_vol)

            # compute output
            output, logits = model(in_vol)

            nll_loss = self.loss_coefficient["NLLLoss"] * criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
            lovasz_loss = self.loss_coefficient["Lovasz_softmax"] * self.ls(output, proj_labels.long())
            # print(f'nll_loss = {nll_loss}')
            # print(f'lovasz_loss = {lovasz_loss}')

            if model_old is not None:
                loss_m = nll_loss + lovasz_loss + \
                    self.loss_coefficient["Distill"] * self.distill(logits, logits_old)
            else:
                loss_m = nll_loss + lovasz_loss

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
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                mask_np = proj_mask[0].cpu().numpy()
                depth_np = in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                mask_np = proj_mask[1].cpu().numpy()
                depth_np = in_vol[1][0].cpu().numpy()
                pred_np = argmax[1].cpu().numpy()
                gt_np = proj_labels[1].cpu().numpy()
                out2 = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                out = np.concatenate([out, out2], axis=0)
                cv2.imshow("sample_training", out)
                cv2.waitKey(1)


            if i % self.ARCH["train"]["report_batch"] == 0:
                log_txt = f"""\
Lr: {lr:.3e} | \
Update: {update_mean:.3e} mean, {update_std:.3e} std | \
Epoch: [{epoch}][{i}/{len(train_loader)}] | \
Time {self.batch_time_t.val:.3f} ({self.batch_time_t.avg:.3f}) | \
Data {self.data_time_t.val:.3f} ({self.data_time_t.avg:.3f}) | \
Loss {losses.val:.4f} ({losses.avg:.4f}) | \
acc {acc.val:.3f} ({acc.avg:.3f}) | \
IoU {iou.val:.3f} ({iou.avg:.3f}) | [{self.calculate_estimate(epoch, i)}]"""
                print(log_txt)

                save_to_log(self.log, 'log.txt', log_txt)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg,hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
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
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(val_loader):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                output, logits = model(in_vol)
                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(),in_vol.size(0))

                wces.update(wce.mean().item(),in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                               mask_np,
                                               pred_np,
                                               gt_np,
                                               color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            if self.uncertainty:
                print('Validation set:\n'       
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Hetero avg {hetero.avg}:.4f\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     hetero=hetero_l,
                                                     acc=acc, iou=iou))
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    self.info["valid_classes/"+class_func(i)] = jacc
            else:

                print('Validation set:\n'
                      'Time avg per batch {batch_time.avg:.3f}\n'
                      'Loss avg {loss.avg:.4f}\n'
                      'Jaccard avg {jac.avg:.4f}\n'
                      'WCE avg {wces.avg:.4f}\n'
                      'Acc avg {acc.avg:.3f}\n'
                      'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                     loss=losses,
                                                     jac=jaccs,
                                                     wces=wces,
                                                     acc=acc, iou=iou))

                save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                                 'Time avg per batch {batch_time.avg:.3f}\n'
                                                 'Loss avg {loss.avg:.4f}\n'
                                                 'Jaccard avg {jac.avg:.4f}\n'
                                                 'WCE avg {wces.avg:.4f}\n'
                                                 'Acc avg {acc.avg:.3f}\n'
                                                 'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                                loss=losses,
                                                                                jac=jaccs,
                                                                                wces=wces,
                                                                                acc=acc, iou=iou))
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc))
                    self.info["valid_classes/" + class_func(i)] = jacc


        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg