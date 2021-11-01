"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@Time: 2021/1/21 3:10 PM
"""


import glob
import gzip
import h5py
import numpy as np
import os
import sys
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

sys.path.append(f"{os.path.dirname(__file__)}/../../..")
from utils.proteins import *

# change this to your data root
DATA_DIR = f'{os.path.dirname(__file__)}/../../../structure_files/atom_sites'
POINT_CLOUD_DIR = os.path.join(DATA_DIR, "point_clouds")
POINT_CLOUD_HDF5 = lambda x: f"{POINT_CLOUD_DIR}/{x}_protein_point_clouds.hdf5"

EVAL_PCT = 0.1


@dataclass
class ProteinPointCloud:
    protein_id: str
    atom_sites: np.ndarray
    
    def labels_to_multihot(self, labels, N):
        arr = np.zeros((N,))
        idx = np.array(labels, dtype=int)
        arr[idx] = 1.0
        return arr

    def get_w_label(self, label_lookup_dict, n_categories:int=18):
        return ProteinPointCloudWLabel(
            protein_id=self.protein_id,
            atom_sites=self.atom_sites,
            labels=self.labels_to_multihot(label_lookup_dict.get(self.protein_id), n_categories))


@dataclass
class ProteinPointCloudWLabel:
    protein_id: str
    atom_sites: np.ndarray
    labels: np.ndarray


def store_point_clouds_w_labels_as_hd5f(point_clouds_w_labels, partition, n_categories):
    if not os.path.exists(POINT_CLOUD_DIR): os.mkdir(POINT_CLOUD_DIR)
    N = len(point_clouds_w_labels)
    with h5py.File(POINT_CLOUD_HDF5(partition), "w") as f:
        f.create_dataset("protein_id", shape=(N,), dtype=h5py.string_dtype(),
                         data=[p.protein_id for p in point_clouds_w_labels])
        f.create_dataset("labels", shape=(N, n_categories), dtype=np.dtype(float),
                         data=[p.labels for p in point_clouds_w_labels])
        f.create_dataset("atom_sites", shape=(N,2048,3), dtype=np.dtype(float),
                         data=np.vstack([p.atom_sites for p in point_clouds_w_labels]))


def read_point_clouds_w_labels_as_hd5f(partition):
    with h5py.File(POINT_CLOUD_HDF5(partition), "r+") as f:
        return f["atom_sites"][:], f["labels"][:]


def protein_to_point_cloud(atom_sites: np.ndarray, num_points: int):
    if atom_sites.shape[0] <= num_points:
        padded = np.concatenate([atom_sites, np.zeros((num_points - atom_sites.shape[0], 3))])
        assert(padded.shape == (num_points, 3))
        return padded
    else:
        idx = np.sort(np.random.choice(np.arange(0, atom_sites.shape[0]), size=num_points, replace=False))
        sampled = atom_sites[idx,:]
        assert (sampled.shape == (num_points, 3))
        return sampled


def create_protein_point_clouds(num_points=2048, overwrite=False):
    dir_exists = os.path.exists(POINT_CLOUD_DIR)
    if dir_exists and overwrite:
        os.rmdir(POINT_CLOUD_DIR)
    if not dir_exists:
        np.random.seed(20211020)  # set a seed to randomize train and eval groups

        label_lookup_dict = load_labels()
        n_categories = int(max([max(k) for k in label_lookup_dict.values() if len(k) > 0]) + 1)

        train_point_clouds = []
        test_point_clouds = []

        atom_files = glob.glob(os.path.join(DATA_DIR, "atom_sites_part_*.parquet"))
        for filename in atom_files:
            print(filename.split("/")[-1])
            atom_sites = pd.read_parquet(filename)
            for id, group in atom_sites.groupby("protein_id"):
                group = protein_to_point_cloud(group[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(pd.to_numeric).to_numpy(), num_points=num_points)
                if np.random.random() < EVAL_PCT:
                    test_point_clouds.append(ProteinPointCloud(id, group))
                else:
                    train_point_clouds.append(ProteinPointCloud(id, group))

        train_point_clouds = [p.get_w_label(label_lookup_dict, n_categories) for p in train_point_clouds if
                              len(label_lookup_dict.get(p.protein_id)) > 0]
        test_point_clouds = [p.get_w_label(label_lookup_dict, n_categories) for p in test_point_clouds if
                              len(label_lookup_dict.get(p.protein_id)) > 0]
        print(f"Number of protein point clouds generated: {len(train_point_clouds) + len(test_point_clouds)}")
        print(f"Number of training proteins: {len(train_point_clouds)}")
        print(f"Number of test proteins: {len(test_point_clouds)}")
        print(f"Storing protein point clouds in hd5f format: {POINT_CLOUD_HDF5}")
        store_point_clouds_w_labels_as_hd5f(train_point_clouds, "train", n_categories)
        store_point_clouds_w_labels_as_hd5f(test_point_clouds, "test", n_categories)
        print("Done created point clouds")

def load_labels():
    def get_index_of_function_label(function, function_arr):
        return function_arr.get(function)

    functions = pd.read_parquet(f"{DATA_DIR}/alphafold_protein_to_parent_gomf_only.parquet")
    functions["parent_gomf_w_name"] = functions.parent_gomf + "|" + functions.parent_gomf_name
    unique_functions = sorted(functions.parent_gomf_w_name.dropna().unique())

    functions_to_index = {function: int(i) for i, function in enumerate(unique_functions)}
    functions['function_idx'] = functions.parent_gomf_w_name.apply(
        lambda x: get_index_of_function_label(x, functions_to_index)
    )
    protein_to_function_labels = functions.groupby("protein_id").function_idx.agg(lambda x: sorted(list(set(x.dropna()))))
    return protein_to_function_labels.to_dict()


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Proteins(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.data, self.label = self.load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    @staticmethod
    def load_data_cls(partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(partition)
        # for f in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        #     f = h5py.File(h5_name, 'r+')
        #     data = f['data'][:].astype('float32')
        #     label = f['label'][:].astype('int64')
        #     f.close()
        #     all_data.append(data)
        #     all_label.append(label)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
