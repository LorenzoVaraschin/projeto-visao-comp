from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import numpy as np
import pandas as pd


class SkyVideoDataset(Dataset):
    def __init__(self, data):
        self.src_paths = list(data["src_paths"])
        self.tgt_paths = list(data["tgt_paths"])
        
        self.src_ghi = list(data["src_ghi"])
        self.tgt_ghi = list(data["tgt_ghi"])

        self.src_ghi_solis = list(data["src_ghi_solis"])
        self.tgt_ghi_solis = list(data["tgt_ghi_solis"])

        self.src_kt_solis = list(data["src_kt_solis"])
        self.tgt_kt_solis = list(data["tgt_kt_solis"])
        
    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, index):
        src_imgs = np.zeros((5, 128, 128, 3))
        tgt_imgs = np.zeros((5, 128, 128, 3))
        for i, (src_path, tgt_path) in enumerate(zip(self.src_paths[index], self.tgt_paths[index])):
            src_imgs[i] = np.load(src_path)
            tgt_imgs[i] = np.load(tgt_path)
            
        src_imgs = torch.tensor(src_imgs).type(torch.float32) / 255
        tgt_imgs = torch.tensor(tgt_imgs).type(torch.float32) / 255
        
        src_imgs = src_imgs.permute(0, 3, 1, 2)
        tgt_imgs = tgt_imgs.permute(0, 3, 1, 2)

        src_ghi, tgt_ghi = torch.tensor(self.src_ghi[index]).type(torch.float32), torch.tensor(self.tgt_ghi[index]).type(torch.float32)

        src_ghi_solis, tgt_ghi_solis = torch.tensor(self.src_ghi_solis[index]).type(torch.float32), torch.tensor(self.tgt_ghi_solis[index]).type(torch.float32)

        src_kt_solis, tgt_kt_solis = torch.tensor(self.src_kt_solis[index]).type(torch.float32), torch.tensor(self.tgt_kt_solis[index]).type(torch.float32)
        
        return src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis

def make_dataloaders(batch_size):
    root_dir = "/home/jovyan/arquivos/solar_forecasting/"
    df_train = pd.read_parquet(root_dir+"train_dataset.parquet.gzip")
    df_val = pd.read_parquet(root_dir+"val_dataset.parquet.gzip")
    df_test = pd.read_parquet(root_dir+"test_dataset.parquet.gzip")
    
    train_dataset = SkyVideoDataset(df_train)
    val_dataset = SkyVideoDataset(df_val)
    test_dataset = SkyVideoDataset(df_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    return train_dataloader, val_dataloader, test_dataloader
