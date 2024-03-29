import json
import os
from socketserver import DatagramRequestHandler
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision
import pathlib
from tasks.config import salsanext

import torch
import math
import random
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings

from tasks.semantic.task import get_task_labels, get_per_task_classes
import tqdm
from tasks.config.dataset import sensor

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

SCAN_FOLDER = 'velodyne'
LABEL_FOLDER = 'labels'


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def sequence_to_string(seq: int):
    return f'{int(seq):02d}'


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data, dim=0)
    project_mask = torch.stack(project_mask, dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment = (proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat(
        (to_augment_unique_5, to_augment_unique_8, to_augment_unique_12), dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat(
            (data, torch.flip(data[k.item()], [2]).unsqueeze(0)), dim=0)
        proj_labels = torch.cat((proj_labels, torch.flip(
            proj_labels[k.item()], [1]).unsqueeze(0)), dim=0)
        project_mask = torch.cat((project_mask, torch.flip(
            project_mask[k.item()], [1]).unsqueeze(0)), dim=0)

    return data, project_mask, proj_labels

TRACK_OUTPUT_DIR = "./aot-benchmark/demo_new_output"

def count_label_numbers(label_files, is_verbose=False):
    if is_verbose:
        print(f"label_files = {label_files}")
    labels_count_dict = {}
    print(f"counting {len(label_files)} labels")
    for label_file in tqdm.tqdm(label_files):
        label = np.fromfile(label_file, dtype=np.int32)
        sem_label = label & 0xFFFF
        values, counts = np.unique(sem_label, return_counts=True)
        for i, v in enumerate(values):
            if v > 259:
                continue
            if v in labels_count_dict:
                labels_count_dict[v] += counts[i]
            else:
                labels_count_dict[v] = counts[i]
    labels_count_dict = {k: v for k, v in sorted(
        labels_count_dict.items(), key=lambda item: item[0])}
    print(f'labels_count_dict = {labels_count_dict}')
    return labels_count_dict


def get_label_frequencies(labels_count_dict: Dict):
    labels_counts_sum = sum(labels_count_dict.values())
    print(f'labels_counts_sum = {labels_counts_sum}')
    labels_frequencies = {key: (float(value) / float(labels_counts_sum))
                          for key, value in labels_count_dict.items()}
    print(f'labels_frequencies = {labels_frequencies}')
    return labels_frequencies


class SemanticKitti(Dataset):

    def __init__(self, root,    # directory where data is
                 sequences,     # sequences for this data (e.g. [1,3,4,6])
                 labels,        # label dict: (e.g 10: "car")
                 color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,    # inverse of previous (recover labels)
                 #  sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True,
                 transform=False,  # send ground truth
                 is_count_labels=False,
                 ):
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        print(f"{type(self).__name__}'s sequences are {self.sequences}")
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        # self.sensor = sensor
        self.sensor_img_H = sensor.img_prop.height
        self.sensor_img_W = sensor.img_prop.width
        self.sensor_img_means = torch.tensor(sensor.img_means,
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor.img_stds,
                                            dtype=torch.float)
        self.sensor_fov_up = sensor.fov_up
        self.sensor_fov_down = sensor.fov_down
        self.max_points = max_points
        self.gt = gt
        self.transform = transform

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print(f"Sequences folder exists! Using sequences from {self.root}")
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert(isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert(isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert(isinstance(self.learning_map, dict))

        # make sure sequences is a list
        assert(isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        self.get_scan_label_files_from_file()
        print(
            f"{type(self).__name__} Using {len(self.scan_files)} scans from sequences {self.sequences}")

        if is_count_labels:
            labels_count_dict = count_label_numbers(self.label_files)
            self.labels_frequencies = get_label_frequencies(labels_count_dict)

    def get_scan_label_files_from_file(self):
        print(f"{type(self).__name__} getting scan and label files")
        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = sequence_to_string(seq)

            print(f"parsing seq {seq}")

            # get paths for each
            scan_path = os.path.join(self.root, seq, SCAN_FOLDER)
            label_path = os.path.join(self.root, seq, LABEL_FOLDER)

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            # check all scans have labels
            if self.gt:
                assert(len(scan_files) == len(label_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()

    def __getitem__(self, index):
        # get item in tensor shape
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # open a semantic laserscan
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                DA=DA,
                                flip_sign=flip_sign,
                                drop_points=drop_points)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down,
                             DA=DA,
                             rot=rot,
                             flip_sign=flip_sign,
                             drop_points=drop_points)

        # open and obtain scan
        scan.open_scan(scan_file)
        if self.gt:
            if label_file.startswith(TRACK_OUTPUT_DIR):
                origin_class_map_to_16_4 = np.array([
                    0, 16, 1, 2, 3, 4, 17, 18, 19, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                ], dtype=np.int32)

                scan.open_label(label_file)
                scan.sem_label = origin_class_map_to_16_4[scan.sem_label]
                scan.proj_sem_label = origin_class_map_to_16_4[scan.proj_sem_label]
            else:
                scan.open_label(label_file)
                # map unused classes to used classes (also for projection)
                scan.sem_label = self.map(scan.sem_label, self.learning_map)
                scan.proj_sem_label = self.map(
                    scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full(
            [self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full(
                [self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        # return
        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]


def read_sample_file(json_file_name: str, sequences: List[str], sample_number: int = None):
    with open(json_file_name, 'r') as f:
        label_file_dict = json.load(f)
    label_files = []
    for seq in sequences:
        seq = sequence_to_string(seq)
        label_files.extend([(seq, file) for file in label_file_dict[seq]])

    if sample_number is not None and (sample_number < len(label_files)):
        label_files = random.sample(label_files, sample_number)

    scan_files = [f"{seq}/{SCAN_FOLDER}/{file.replace(EXTENSIONS_LABEL[0], EXTENSIONS_SCAN[0])}"
                  for seq, file in label_files]
    label_files = [f"{seq}/{LABEL_FOLDER}/{file}"
                   for seq, file in label_files]
    return scan_files, label_files


def read_sample_file_txt(txt_file_name: str, sequences: List[str], sample_number: int = None):
    with open(txt_file_name, 'r') as f:
        lines = f.readlines()
    label_file_lines = lines

    if sample_number is not None and (sample_number < len(label_file_lines)):
        label_file_lines = random.sample(label_file_lines, sample_number)

    scan_files = [
        line.split()[1] for line in label_file_lines]
    label_files = [
        line.split()[2] for line in label_file_lines]
    return scan_files, label_files


class IncrementalSemanticKitti(SemanticKitti):

    def __init__(self, root,    # directory where data is
                 sequences,     # sequences for this data (e.g. [1,3,4,6])
                 labels,        # label dict: (e.g 10: "car")
                 color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,    # inverse of previous (recover labels)
                 #  sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True,
                 transform=False,  # send ground truth
                 sample_number: int = None,
                 is_count_labels=False,
                 extend_num=0,
                 ):
        self.sample_number = sample_number
        self.extend_num = extend_num
        super(IncrementalSemanticKitti, self).__init__(
            root=root,
            sequences=sequences,
            labels=labels,
            color_map=color_map,
            learning_map=learning_map,
            learning_map_inv=learning_map_inv,
            # sensor=sensor,
            max_points=max_points,
            gt=gt,
            transform=transform,
            is_count_labels=is_count_labels,
        )

    def get_scan_label_files_from_file(self):
        print(f"{type(self).__name__} getting scan and label files")
        # fill in with names, checking that all sequences are complete
        samplefiles = [
            "motorcyclist",
            "bicyclist", "car", "person",
        ]
        samples = {}
        for name in samplefiles:
            txt_file_path = os.path.join(self.root, f"{name}.txt")
            print(f"reading {txt_file_path}")
            scan_files, label_files = read_sample_file_txt(
                txt_file_path, self.sequences, self.sample_number)
            scan_files = [os.path.join(self.root, scan)
                          for scan in scan_files]
            label_files = [os.path.join(self.root, label)
                           for label in label_files]
            track_total_num = 15 * 10 + 1
            while True:
                print(f"{scan_files = }")
                pass_check = True
                for scanfile in scan_files:
                    scanfile = pathlib.Path(scanfile)
                    scanfile_stem = scanfile.stem
                    scanfile_stem = int(scanfile_stem)
                    scanfile_seq = scanfile.parents[1].stem
                    scanfile_seq_dir = f"/public/home/zhoujunbao/datasets/semantickitti/sequences/{scanfile_seq}/labels"
                    scanfile_seq_len = len(os.listdir(scanfile_seq_dir))
                    print(f"{scanfile_seq_len = }")
                    if (scanfile_stem < track_total_num) or ((scanfile_seq_len - scanfile_stem) < track_total_num):
                        pass_check = False
                        break
                if pass_check:
                    break
                else:
                    print(f"Resample !")
                    scan_files, label_files = read_sample_file_txt(
                        txt_file_path, self.sequences, self.sample_number)
                    scan_files = [os.path.join(self.root, scan)
                                for scan in scan_files]
                    label_files = [os.path.join(self.root, label)
                                for label in label_files]


            # count_label_numbers(label_files, is_verbose=False)
            if name not in samples:
                samples[name] = [scan_files, label_files]

        for _, (scan_files, label_files) in samples.items():
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()

        print(f"sampled {self.scan_files = }")
        print(f"sampled {self.label_files = }")

        if self.extend_num > 0:
            # os.system(f"rm -r {track_output_dir}")
            track_total_num = salsanext.train.extend_gap * self.extend_num + 1
            
            all_extend_scan_files = []
            all_extend_label_files = []
            for scanfile in self.scan_files:
                scanfile = pathlib.Path(scanfile)
                print(f"{scanfile = }")
                start_index = int(scanfile.stem)
                sequence_name = scanfile.parents[1].stem
                print(f"{sequence_name = }")

                track_frame_index = list(range(start_index, start_index+salsanext.train.extend_gap * self.extend_num+1, salsanext.train.extend_gap))
                track_frame_index.remove(start_index)
                track_frame_index += list(range(start_index, start_index-salsanext.train.extend_gap * self.extend_num-1, -salsanext.train.extend_gap))
                track_frame_index.remove(start_index)

                print(f"{track_frame_index = }")
                track_frame_index = [str(idx).zfill(6) for idx in track_frame_index]
                print(f"{track_frame_index = }")

                extend_scan_files = [
                    os.path.join(self.root, sequence_name, SCAN_FOLDER, f"{idx}.bin")
                    for idx in track_frame_index
                ]
                print(f"{extend_scan_files = }")
                all_extend_scan_files += extend_scan_files

                extend_label_files = [
                    os.path.join(TRACK_OUTPUT_DIR, sequence_name, f"start_at_{start_index}", "pred_sem_labels_only_novel", f"{idx}.label")
                    for idx in track_frame_index
                ]
                print(f"{extend_label_files = }")
                all_extend_label_files += extend_label_files

                if salsanext.train.is_generate_extend_frame:
                    os.system(
                        f"""
    cd aot-benchmark;
    python tools/demo_kitti.py --seq_name {sequence_name} --start_index {start_index} --max_track_frame_num {track_total_num}
    """)
                    os.system(
                        f"""
    cd aot-benchmark;
    python tools/demo_kitti.py --seq_name {sequence_name} --start_index {start_index} --max_track_frame_num {track_total_num} --is_reverse_seq
    """)
            self.scan_files += all_extend_scan_files
            self.label_files += all_extend_label_files
            print(f"After extend {self.scan_files = }")
            print(f"After extend {self.label_files = }")


class Parser():
    def __init__(self,
                 root,              # directory for data
                 #  datargs,
                 #  archargs,
                 batch_size,
                 is_test,
                 gt=True,           # get gt?
                 shuffle_train=True):  # shuffle training set?
        super(Parser, self).__init__()

        print(10 * '-', '\n', f'Initializing {type(self).__name__}')

        # if I am training, get the dataset
        self.root = root
        # self.DATA = datargs
        # self.ARCH = archargs

        from tasks.config import dataset
        from tasks.config import semantic_kitti
        from tasks.config import salsanext

        self.train_sequences = semantic_kitti.split.train
        self.valid_sequences = semantic_kitti.split.valid
        self.test_sequences = semantic_kitti.split.test

        self.labels = semantic_kitti.labels
        self.color_map = semantic_kitti.color_map
        self.learning_map = semantic_kitti.learning_map
        print(f"{type(self).__name__}.learning_map = {self.learning_map}")
        self.label_frequencies = semantic_kitti.label_frequencies
        self.learning_map_inv = semantic_kitti.learning_map_inv
        # self.sensor = dataset.sensor
        self.max_points = dataset.max_points
        self.batch_size = batch_size
        # self.workers = dataset.workers

        self.task_name = salsanext.train.task_name
        self.task_step = salsanext.train.task_step

        self.gt = gt
        self.shuffle_train = shuffle_train

        self.update_class_map()
        print(f"{type(self).__name__}.learning_map = {self.learning_map}")

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)
        self.xentropy_label_frequencies = self.label_frequencies_to_xentropy(
            self.label_frequencies)

        if not is_test:
            if self.task_step > 0:
                self.train_dataset = IncrementalSemanticKitti(
                    root=self.root,
                    sequences=self.train_sequences,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    # sensor=self.sensor,
                    max_points=self.max_points,
                    transform=True,
                    gt=self.gt,
                    sample_number=salsanext.train.sample_number,
                    is_count_labels=True,
                    extend_num=salsanext.train.extend_num,
                )
                self.xentropy_label_frequencies = self.label_frequencies_to_xentropy(
                    self.train_dataset.labels_frequencies)
            else:
                # Data loading code
                self.train_dataset = SemanticKitti(
                    root=self.root,
                    sequences=self.train_sequences,
                    labels=self.labels,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    # sensor=self.sensor,
                    max_points=self.max_points,
                    transform=True,
                    gt=self.gt)

            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=min(len(self.train_dataset), self.batch_size),
                shuffle=self.shuffle_train,
                num_workers=dataset.workers,
                drop_last=True)
            assert len(self.trainloader) > 0
            self.trainiter = iter(self.trainloader)

            self.valid_dataset = SemanticKitti(
                root=self.root,
                sequences=self.valid_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                # sensor=self.sensor,
                max_points=self.max_points,
                gt=self.gt)

            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=dataset.workers,
                drop_last=True)
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if is_test and self.test_sequences:
            self.test_dataset = SemanticKitti(
                root=self.root,
                sequences=self.test_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                # sensor=self.sensor,
                max_points=self.max_points,
                gt=False)

            self.testloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=dataset.workers,
                drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def update_class_map(self):
        print(f"Updating labels")

        labels_new, labels_old = get_task_labels(
            self.task_name, self.task_step)
        labels = labels_old + labels_new
        print(f"labels = {labels}")
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i
        print(f"label_map = {label_map}")

        # update learning_map
        learning_map_new = self.learning_map.copy()
        for key, value in learning_map_new.items():
            if value not in labels:
                self.learning_map[key] = 0
            else:
                self.learning_map[key] = label_map[value]

        # update learning_map_inv
        learning_map_inv_new = sorted(
            self.learning_map.items(), key=lambda x: x[1], reverse=False)
        self.learning_map_inv = {}
        for (key, value) in learning_map_inv_new:
            if value not in self.learning_map_inv.keys():
                self.learning_map_inv[value] = key

        # for key, value in self.learning_map_inv.items():
        #   print(f"{key} -> {self.labels[value]}")
        # pass

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)

    def label_frequencies_to_xentropy(self, label_frequencies):
        result = torch.zeros(self.get_n_classes(), dtype=torch.float)
        for label, freq in label_frequencies.items():
            # map actual class to xentropy class
            # print(f"label = {label}")
            x_label = self.to_xentropy(label)
            # print(f"x_label = {x_label}")
            result[x_label] += freq
        return result


if __name__ == "__main__":
    pass
    # parser = Parser(
    #     root="/public/home/meijilin/dataset/semantickitti",
    #     # datargs=self.DATA,
    #     # archargs=self.ARCH,
    #     batch_size=32,
    #     is_test=False,
    #     gt=True,
    #     shuffle_train=True,
    # )
