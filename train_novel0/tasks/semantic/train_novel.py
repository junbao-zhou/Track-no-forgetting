#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import shutil
from shutil import copyfile, copytree
import __init__ as booger

import sys

import os.path as osp

# TRAIN_PATH = "/home/user/ANS/Experiments/SalsaNext-master/train_base0"
# sys.path.insert(0, TRAIN_PATH)

import yaml
from tasks.semantic.modules.trainer import *
from pip._vendor.distlib.compat import raw_input

from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.SalsaNext import *
#from tasks.semantic.modules.save_dataset_projected import *
import math
from decimal import Decimal
from utils import str2bool, Tee

def remove_exponent(d):
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    if prefixes:
        millnames = ['']
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    result = '{:.{precision}f}'.format(n / 10**(3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return '{0}{dx}'.format(result, dx=millnames[millidx])


current_file_path = pathlib.Path(__file__)
project_folder = current_file_path.parent.parent.parent.resolve()

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train_novel.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default="/public/home/meijilin/dataset/semantickitti",
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=project_folder,
        required=False,
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default="novel",
        help='If you want to give an aditional discriptive name'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default="",
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--manual_seed', '-ms',
        type=int,
        required=False,
        default=123,
        help='Manual random seed for reproducity'
    )
    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = os.path.join(FLAGS.log, 'logs', datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-") + FLAGS.name)

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("pretrained", FLAGS.pretrained)
    print("----------\n")
    # print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # create log folder
    try:
        if FLAGS.pretrained == "":
            FLAGS.pretrained = None
            if os.path.isdir(FLAGS.log):
                if os.listdir(FLAGS.log):
                    answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
        else:
            FLAGS.log = FLAGS.pretrained
            print("Not creating new log file. Using pretrained directory")
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()
    sys.stdout = Tee(os.path.join(FLAGS.log, "print.log"))

    # does model folder exist?
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print("model folder exists! Using model from %s" % (FLAGS.pretrained))
        else:
            print("model folder doesnt exist! Start with random weights...")
    else:
        print("No pretrained directory found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print(f"Copying files to {FLAGS.log} for further reference.")
        copytree(
            current_file_path.parent.parent, os.path.join(FLAGS.log, "codes"),
            ignore=lambda src, children: [child for child in children if '__pycache__' in child])
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    set_random_seed(FLAGS.manual_seed)
    # create trainer and start the training
    torch.set_printoptions(linewidth=400)
    trainer = Trainer(
        # ARCH, DATA,
        datadir=FLAGS.dataset, logdir=FLAGS.log, path=None)
    trainer.train()
