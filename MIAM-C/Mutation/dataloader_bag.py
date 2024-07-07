import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, glob, torch, random, itertools
from PIL import Image

class TrainDataset_Bag(Dataset):
    def __init__(self, file_dir, transform, bag_size):
        bg_list = glob.glob(file_dir+"/WILD/*/*.jpg")
        tumor_list = glob.glob(file_dir+"/BRCA/*/*.jpg")#换任务的话，需要修改下
        random.shuffle(bg_list)
        random.shuffle(tumor_list)
        bg_list_chunks = [bg_list[x:x+bag_size] for x in range(0, len(bg_list)-bag_size, bag_size)]
        tumor_list_chunks = [tumor_list[x:x+bag_size] for x in range(0, len(tumor_list)-bag_size, bag_size)]
        self.list_chunks = tumor_list_chunks + bg_list_chunks
        random.shuffle(self.list_chunks)
        self.transform = transform
        self.bag_size = bag_size

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        bag_image = torch.zeros(self.bag_size, 3, 256, 256)
        wsi_type = 1 if 'BRCA' in file_chunk[0] else 0
        # print(len(file_chunk))
        for num in range(self.bag_size):
            file_name = self.list_chunks[idx][num]
            # print(file_name)
            image = Image.open(file_name)
            image = self.transform(image)
            bag_image[num,:,:,:] = image
        return bag_image, wsi_type

class TrainDataset_Fold_Bag(Dataset):#5折用dataloader
    def __init__(self, file_dirs, transform, bag_size, ratio = 1):
        wild_list_=[]
        for file_dir in file_dirs[0]:
            pic_list = glob.glob(file_dir+"/*.jpg")
            random.shuffle(pic_list)
            wild_list_.append(pic_list)
        wild_list = list(itertools.chain.from_iterable(wild_list_))
        brca_list_=[]
        for file_dir in file_dirs[1]:
            pic_list = glob.glob(file_dir+"/*.jpg")
            random.shuffle(pic_list)
            brca_list_.append(pic_list)
        brca_list = list(itertools.chain.from_iterable(brca_list_))
        # random.shuffle(wild_list)
        # random.shuffle(brca_list)
        wild_list_chunks = [wild_list[x:x+bag_size] for x in range(0, len(wild_list)-bag_size, bag_size)]
        brca_list_chunks = [brca_list[x:x+bag_size] for x in range(0, len(brca_list)-bag_size, bag_size)]
        self.list_chunks = brca_list_chunks + wild_list_chunks
        # random.shuffle(self.list_chunks)
        self.transform = transform
        self.bag_size = bag_size
        self.list_chunks = self.list_chunks[:int(len(self.list_chunks)*ratio)]
        # print('length',len(self.list_chunks))

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        bag_image = torch.zeros(self.bag_size, 3, 256, 256)
        wsi_type = 1 if 'BRCA' in file_chunk[0] else 0
        # print(len(file_chunk))
        for num in range(len(self.list_chunks[idx])):
            file_name = self.list_chunks[idx][num]
            # print(file_name)
            image = Image.open(file_name)
            image = self.transform(image)
            bag_image[num,:,:,:] = image
        return bag_image, wsi_type

class TestDataset(Dataset):
    def __init__(self, wsi_dir, transform, bag_size):
        self.wsi_dir = wsi_dir
        self.bag_size = bag_size
        pic_list = glob.glob(self.wsi_dir+'/*.jpg')
        # random.shuffle(pic_list)
        self.list_chunks = [pic_list[x:x+bag_size] for x in range(0, len(pic_list)-bag_size, bag_size)]
        # random.shuffle(self.list_chunks)
        self.transform = transform

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        bag_image = torch.zeros(self.bag_size, 3, 256, 256)
        for num in range(self.bag_size):
            file_name = self.list_chunks[idx][num]
            image = Image.open(file_name)
            image = self.transform(image)
            bag_image[num,:,:,:] = image
        return bag_image

class ValDataset(Dataset):
    def __init__(self, file_dirs, transform, bag_size):
        wild_list_=[]
        for file_dir in file_dirs[0]:
            wild_list_.append(glob.glob(file_dir+"/*.jpg"))
        wild_list = list(itertools.chain.from_iterable(wild_list_))
        brca_list_=[]
        for file_dir in file_dirs[1]:
            brca_list_.append(glob.glob(file_dir+"/*.jpg"))
        brca_list = list(itertools.chain.from_iterable(brca_list_))
        random.shuffle(wild_list)
        random.shuffle(brca_list)
        wild_list_chunks = [wild_list[x:x+bag_size] for x in range(0, len(wild_list)-bag_size, bag_size)]
        brca_list_chunks = [brca_list[x:x+bag_size] for x in range(0, len(brca_list)-bag_size, bag_size)]
        self.list_chunks = wild_list_chunks + brca_list_chunks
        random.shuffle(self.list_chunks)
        self.transform = transform
        self.bag_size = bag_size
        print('length',len(self.list_chunks))

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        file_chunk = self.list_chunks[idx]
        bag_image = torch.zeros(self.bag_size, 3, 256, 256)
        wsi_type = 1 if 'BRCA' in file_chunk[0] else 0
        # print(len(file_chunk))
        for num in range(len(self.list_chunks[idx])):
            file_name = self.list_chunks[idx][num]
            # print(file_name)
            image = Image.open(file_name)
            image = self.transform(image)
            bag_image[num,:,:,:] = image
        return bag_image, wsi_type