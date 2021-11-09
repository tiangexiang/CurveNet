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
BASE_POINT_CLOUD_DIR = os.path.join(DATA_DIR, "point_clouds")
POINT_CLOUD_HDF5 = lambda x, y: f"{POINT_CLOUD_DIR(x)}/{y}_protein_point_clouds.hdf5"

EVAL_PCT = 0.1


def get_point_cloud_dir(name):
    return os.path.join(BASE_POINT_CLOUD_DIR, name)


def get_point_cloud_hdf5(name, train_test):
    return f"{get_point_cloud_dir(name)}/{train_test}_protein_point_clouds.hdf5"


@dataclass
class ProteinPointCloud:
    protein_id: str
    atom_sites: np.ndarray
    num_atoms: int
    
    def labels_to_multihot(self, labels, N):
        arr = np.zeros((N,))
        idx = np.array(labels, dtype=int)
        arr[idx] = 1.0
        return arr

    def get_w_label(self, label_lookup_dict, n_categories:int=18):
        return ProteinPointCloudWLabel(
            protein_id=self.protein_id,
            atom_sites=self.atom_sites,
            num_atoms=self.num_atoms,
            labels=self.labels_to_multihot(label_lookup_dict.get(self.protein_id), n_categories))


@dataclass
class ProteinPointCloudWLabel:
    protein_id: str
    atom_sites: np.ndarray
    num_atoms: int
    labels: np.ndarray


def store_point_clouds_w_labels_as_hd5f(name, point_clouds_w_labels, partition, num_points, n_categories):
    POINT_CLOUD_DIR = get_point_cloud_dir(name)
    
    if not os.path.exists(POINT_CLOUD_DIR): os.mkdir(POINT_CLOUD_DIR)
    N = len(point_clouds_w_labels)
    with h5py.File(get_point_cloud_hdf5(name, partition), "w") as f:
        points = np.stack([p.atom_sites for p in point_clouds_w_labels])
        print(points.shape)
        print((N,num_points,3))
        f.create_dataset("atom_sites", shape=(N,num_points,3), dtype=np.dtype(float),
                         data=points)
            
        f.create_dataset("protein_id", shape=(N,), dtype=h5py.string_dtype(),
                         data=[p.protein_id for p in point_clouds_w_labels])
        
        f.create_dataset("labels", shape=(N, n_categories), dtype=np.dtype(float),
                         data=[p.labels for p in point_clouds_w_labels])
        
        f.create_dataset("num_atoms", shape=(N,), dtype='i8',
                         data=[p.num_atoms for p in point_clouds_w_labels])
    


def read_point_clouds_w_labels_as_hd5f(name, partition):
    with h5py.File(get_point_cloud_hdf5(name, partition), "r+") as f:
        return f["atom_sites"][:], f["labels"][:]


def one_per_amino_acid(atom_sites: pd.DataFrame):
    return atom_sites[atom_sites["label_atom_id"] == "CA"].reset_index()


def protein_pandas_to_numpy(group):
    return group[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(pd.to_numeric).to_numpy()

    
def mask_by_confidence(atom_group):
    return atom_group[atom_group.confidence_pLDDT.astype(float) > 50].sort_values("id")
    
    
def protein_to_sampled_point_cloud(atom_sites: pd.DataFrame, num_points: int):
    atom_sites = protein_pandas_to_numpy(atom_sites).astype(float)
    if atom_sites.shape[0] <= num_points:
        padded = np.concatenate([atom_sites, np.zeros((num_points - atom_sites.shape[0], 3))])
        assert(padded.shape == (num_points, 3))
        return padded
    else:
        idx = np.sort(np.random.choice(np.arange(0, atom_sites.shape[0]), size=num_points, replace=False))
        sampled = atom_sites[idx,:]
        assert (sampled.shape == (num_points, 3))
        return sampled
    

def protein_to_masked_point_cloud(atom_sites: pd.DataFrame, num_points: int):
    atom_sites = mask_by_confidence(atom_sites)
    return protein_to_sampled_point_cloud(atom_sites, num_points)
    

def create_protein_point_clouds(name, num_points=2048, overwrite=False):
    if not os.path.exists(BASE_POINT_CLOUD_DIR): os.mkdir(BASE_POINT_CLOUD_DIR)
    POINT_CLOUD_DIR = get_point_cloud_dir(name)
    
    dir_exists = os.path.exists(POINT_CLOUD_DIR)
    if dir_exists and overwrite:
        os.rmdir(POINT_CLOUD_DIR)
    if not dir_exists:
        label_lookup_dict = load_labels()
        n_categories = int(max([max(k) for k in label_lookup_dict.values() if len(k) > 0]) + 1)

        train_point_clouds = []
        test_point_clouds = []

        atom_files = glob.glob(os.path.join(DATA_DIR, "atom_sites_part_*.parquet"))
        for filename in atom_files:
            print(filename.split("/")[-1])
            atom_sites = pd.read_parquet(filename)
            for id, group in atom_sites.groupby("protein_id"):
                group = one_per_amino_acid(group)
                group = point_cloud_method_by_name[name](group, num_points=num_points)
                if group.shape[0]==0:
                    print("no points, skipping")
                    continue
                elif hash(id) % 100 < EVAL_PCT*100:
                    test_point_clouds.append(ProteinPointCloud(id, group, group.shape[0]))
                else:
                    train_point_clouds.append(ProteinPointCloud(id, group, group.shape[0]))

        train_point_clouds = [p.get_w_label(label_lookup_dict, n_categories) for p in train_point_clouds if
                              len(label_lookup_dict.get(p.protein_id)) > 0]
        test_point_clouds = [p.get_w_label(label_lookup_dict, n_categories) for p in test_point_clouds if
                              len(label_lookup_dict.get(p.protein_id)) > 0]
        print(f"Number of protein point clouds generated: {len(train_point_clouds) + len(test_point_clouds)}")
        print(f"Number of training proteins: {len(train_point_clouds)}")
        print(f"Number of test proteins: {len(test_point_clouds)}")
        print(f"Storing protein point clouds in hd5f format: {POINT_CLOUD_HDF5}")
        store_point_clouds_w_labels_as_hd5f(name, train_point_clouds, "train", num_points, n_categories)
        store_point_clouds_w_labels_as_hd5f(name, test_point_clouds, "test", num_points, n_categories)
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


class ProteinsSampled(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.num_points = num_points
        self.partition = partition
        self.max_points = 2048
        self.data, self.label = self.load_data_cls(partition)

        
    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name="sampled", num_points=self.max_points, overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(name="sampled", partition=partition)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
    
class ProteinsExtended(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.num_points = num_points
        self.partition = partition
        self.max_points = 2700
        self.data, self.label = self.load_data_cls(partition)

    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name="sequence_head", num_points=self.max_points, overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(name="sequence_head", partition=partition)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

        
class ProteinsExtendedWithMask(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.num_points = num_points
        self.partition = partition
        self.max_points = 4000
        self.data, self.label = self.load_data_cls(partition)
        
    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name="sequence_head_w_confidence_mask", num_points=self.max_points, overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(name="sequence_head_w_confidence_mask", partition=partition)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
    
point_cloud_method_by_name = {
    "sampled": protein_to_sampled_point_cloud,
    "sequence_head": protein_to_sampled_point_cloud,
    "sequence_head_w_confidence_mask": protein_to_masked_point_cloud,
}