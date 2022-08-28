import json
import os
from socketserver import DatagramRequestHandler
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision

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


def count_label_numbers(label_files, is_verbose=False):
    if is_verbose:
        print(f"label_files = {label_files}")
    labels_count_dict = {}
    print(f"counting {len(label_files)} labels")
    for label_file in tqdm.tqdm(label_files):
        label = np.fromfile(label_file, dtype=np.int32)
        values, counts = np.unique(label, return_counts=True)
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
                 sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True,
                 transform=False,  # send ground truth
                 ):
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        print(f"{type(self).__name__}'s sequences are {self.sequences}")
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
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
            f"Using {len(self.scan_files)} scans from sequences {self.sequences}")

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

        labels_count_dict = count_label_numbers(self.label_files)
        self.labels_frequencies = get_label_frequencies(labels_count_dict)

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


class IncrementalSemanticKitti(SemanticKitti):

    def __init__(self, root,    # directory where data is
                 sequences,     # sequences for this data (e.g. [1,3,4,6])
                 labels,        # label dict: (e.g 10: "car")
                 color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,    # inverse of previous (recover labels)
                 sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True,
                 transform=False,  # send ground truth
                 sample_number: int = None,
                 ):
        self.sample_number = sample_number
        super(IncrementalSemanticKitti, self).__init__(
            root,
            sequences,
            labels,
            color_map,
            learning_map,
            learning_map_inv,
            sensor,
            max_points,
            gt,
            transform,
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
            json_file_path = os.path.join(self.root, f"{name}.json")
            print(f"reading {json_file_path}")
            scan_files, label_files = read_sample_file(
                json_file_path, self.sequences, self.sample_number)
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

        labels_count_dict = count_label_numbers(self.label_files)
        self.labels_frequencies = get_label_frequencies(labels_count_dict)


class Parser():
    def __init__(self,
                 root,              # directory for data
                 datargs,
                 archargs,
                 gt=True,           # get gt?
                 shuffle_train=True):  # shuffle training set?
        super(Parser, self).__init__()

        print(10 * '-', '\n', f'Initializing {type(self).__name__}')

        # if I am training, get the dataset
        self.root = root
        self.DATA = datargs
        self.ARCH = archargs

        self.train_sequences = self.DATA["split"]["train"]
        self.valid_sequences = self.DATA["split"]["valid"]
        self.test_sequences = self.DATA["split"]["test"]
        self.labels = self.DATA["labels"]
        self.color_map = self.DATA["color_map"]
        self.learning_map = self.DATA["learning_map"]
        print(f"{type(self).__name__}.learning_map = {self.learning_map}")
        self.label_frequencies = self.DATA["label_frequencies"]
        self.learning_map_inv = self.DATA["learning_map_inv"]
        self.sensor = self.ARCH["dataset"]["sensor"]
        self.max_points = self.ARCH["dataset"]["max_points"]
        self.batch_size = self.ARCH["train"]["batch_size"]
        self.workers = self.ARCH["train"]["workers"]

        self.task_name = self.ARCH["train"]["task_name"]
        self.task_step = self.ARCH["train"]["task_step"]

        self.gt = gt
        self.shuffle_train = shuffle_train

        self.update_class_map()
        print(f"{type(self).__name__}.learning_map = {self.learning_map}")

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)
        self.xentropy_label_frequencies = self.label_frequencies_to_xentropy(
            self.label_frequencies)

        if self.task_step > 0:
            self.train_dataset = IncrementalSemanticKitti(
                root=self.root,
                sequences=self.train_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                sensor=self.sensor,
                max_points=self.max_points,
                transform=True,
                gt=self.gt,
                sample_number=self.ARCH["train"]["sample_number"]
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
                sensor=self.sensor,
                max_points=self.max_points,
                transform=True,
                gt=self.gt)

        self.trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.workers,
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
            sensor=self.sensor,
            max_points=self.max_points,
            gt=self.gt)

        self.validloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = SemanticKitti(
                root=self.root,
                sequences=self.test_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                sensor=self.sensor,
                max_points=self.max_points,
                gt=False)

            self.testloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
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
