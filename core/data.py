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
from utils.proteins import *

# change this to your data root
DATA_DIR = '../../UP000005640_9606_HUMAN/'
POINT_CLOUD_DIR = os.path.join(DATA_DIR, "point_clouds")
CIF_SUFFIX = ".cif.gz"
POINT_CLOUD_HDF5 = f"{POINT_CLOUD_DIR}/protein_point_clouds.hdf5"

@dataclass
class ProteinPointCloud:
    protein_id: str
    atom_sites: np.ndarray

    def get_w_label(self, label_lookup_dict):
        return ProteinPointCloudWLabel(
            protein_id=self.protein_id,
            atom_sites=self.atom_sites,
            labels=label_lookup_dict.get(self.protein_id))

@dataclass
class ProteinPointCloudWLabel:
    protein_id: str
    atom_sites: np.ndarray
    labels: np.ndarray

def store_point_clouds_w_labels_as_hd5f(point_clouds_w_labels):
    N = len(point_clouds_w_labels)
    with h5py.File(POINT_CLOUD_HDF5, "w") as f:
        f.create_dataset("protein_id", shape=(N,), dtype=np.dtype(str),
                         data=[p.protein_id for p in point_clouds_w_labels])
        f.create_dataset("atom_sites", shape=(N, 3), dtype=np.dtype(float),
                         data=[p.protein_id for p in point_clouds_w_labels])
        f.create_dataset("labels", shape=(N,), dtype=np.dtype('O'),
                         data=[p.labels for p in point_clouds_w_labels])

def read_point_clouds_w_labels_as_hd5f():
    with h5py.File(POINT_CLOUD_HDF5, "r") as f:
        return f["atom_sites"], f["labels"]


def protein_to_point_cloud(atom_sites: np.ndarray, num_points: int):
    if atom_sites.shape[0] <= num_points:
        padded = np.concatenate([atom_sites, np.zeros((num_points - atom_sites.shape[0], 3))])
        assert(padded.shape == (num_points, 3))
        return padded
    else:
        idx = np.sort(np.random.choice(np.arange(0, atom_sites.shape[0]), size=num_points, replace=False))
        return atom_sites[idx,:]


def create_protein_point_clouds(overwrite=False):
    dir_exists = os.path.exists(POINT_CLOUD_DIR)
    if dir_exists and overwrite:
        os.rmdir(POINT_CLOUD_DIR)
    if not dir_exists:
        os.mkdir(POINT_CLOUD_DIR)

        label_lookup_dict = load_labels()

        point_clouds = []
        for filename in glob.glob(DATA_DIR, "*"+CIF_SUFFIX):
            protein_id = get_protein_id_from_filename(filename)
            with gzip.open(filename, 'rt') as f:
                atom_sites = get_atom_sites_from_cif(parse_cif(f.read()))[["protein_"]]
                atom_sites = atom_sites[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy()
            point_clouds.append(ProteinPointCloud(protein_id, atom_sites))

        point_clouds = [p.get_w_label(label_lookup_dict) for p in point_clouds]
        store_point_clouds_w_labels_as_hd5f(point_clouds)


def load_labels():
    def get_index_of_function_label(function, function_arr):
        return function_arr.index(function)

    functions = pd.read_parquet(f"{DATA_DIR}/functional_sim_gomf_to_alphafold_protein.parquet")
    functions["parent_gomf_w_name"] = functions.parent_gomf + "|" + functions.parent_gomf_name
    unique_functions = sorted(functions.parent_gomf_w_name.unique())
    functions['function_idx'] = functions.parent_gomf_w_name.apply(
        lambda x: get_index_of_function_label(x, unique_functions)
    )
    protein_to_function_labels = functions.groupby("protein_id").function_idx.agg(lambda x: sorted(list(set(x))))
    return protein_to_function_labels.to_dict()


def load_data_cls(partition, overwrite=False):
    if not os.path.exists(DATA_DIR):
        raise Exception("first download UP000005640_9606_HUMAN into project root")
    create_protein_point_clouds(overwrite=overwrite)
    all_data, all_label = read_point_clouds_w_labels_as_hd5f()
    # for f in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
    #     f = h5py.File(h5_name, 'r+')
    #     data = f['data'][:].astype('float32')
    #     label = f['label'][:].astype('int64')
    #     f.close()
    #     all_data.append(data)
    #     all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


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


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            #pointcloud = rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
