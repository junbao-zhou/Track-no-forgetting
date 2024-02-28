#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import pathlib
import subprocess
import datetime
import yaml
from shutil import copyfile, copytree
import os
import shutil
import __init__ as booger


from tasks.semantic.modules.user import *
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

current_file_path = pathlib.Path(__file__)
project_folder = current_file_path.parent.parent.parent.resolve()

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default="/public/home/meijilin/dataset/semantickitti/dataset",
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=project_folder,
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default="novel",
        help='If you want to give an aditional discriptive name'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default="/public/home/meijilin/zhoujunbao/SalsaNext/train_novel0/logs/2022-08-27-02:59:39novel",
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )


    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default="test",
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = os.path.join(FLAGS.log, 'infer_logs', datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S-") + FLAGS.name)

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # create log folder
    from tasks.config import semantic_kitti
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

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

    # create user and infer dataset
    user = User(
        FLAGS.dataset,
        FLAGS.log,
        FLAGS.model,
        FLAGS.split,
        FLAGS.uncertainty,
        FLAGS.monte_carlo,
    )
    user.infer()
