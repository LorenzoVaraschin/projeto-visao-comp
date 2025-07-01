import lightning as L
import torchmetrics
from torchmetrics import MetricCollection
from aux_modules import TransformerBlock, PositionalEncoding, LinearPatchEncoder, LinearPatchDecoder
import torch
from torch import nn
import math 
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import time
import itertools
from tqdm import tqdm
import torchvision
import segmentation_models_pytorch as smp

class PredFormer(L.LightningModule):
    def __init__(self, h, d_model, d_hidden, N, patch_size, image_size, dropout, num_frames, attn_type, transformer_type, autoencoder_type, sched_max_steps, learning_rate, lr_scheduler):
        super().__init__()
        self.save_hyperparameters()
        self.h = h
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.N = N # number of transformer blocks
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_h = image_size // patch_size
        self.n_w = image_size // patch_size
        self.dropout = dropout
        self.num_frames = num_frames
        self.attn_type = attn_type
        self.transformer_type = transformer_type
        assert self.transformer_type in ["encoder_only", "decoder_only"]
        self.autoencoder_type = autoencoder_type
        assert self.autoencoder_type in ["linear", "unet"]
        self.sched_max_steps = sched_max_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler

        self.encoder_stack = nn.ModuleList([TransformerBlock(h, d_model, d_hidden, attn_type, num_frames, self.n_h*self.n_w, transformer_type, dropout) for _ in range(N)])

        if self.attn_type == "space_time":
            self.encoder_stack = nn.ModuleList([TransformerBlock(h, d_model, d_hidden, attn_type, num_frames, self.n_h*self.n_w, transformer_type, dropout) for _ in range(N)])
        elif self.attn_type == "quad_TSST":
            assert self.N % 4 == 0
            self.encoder_stack = nn.ModuleList([])
            for i in range(0, self.N, 4):
                self.encoder_stack.append(TransformerBlock(h, d_model, d_hidden, "temporal", num_frames, self.n_h*self.n_w, transformer_type, dropout))
                self.encoder_stack.append(TransformerBlock(h, d_model, d_hidden, "spatial", num_frames, self.n_h*self.n_w, transformer_type, dropout))
                self.encoder_stack.append(TransformerBlock(h, d_model, d_hidden, "spatial", num_frames, self.n_h*self.n_w, transformer_type, dropout))
                self.encoder_stack.append(TransformerBlock(h, d_model, d_hidden, "temporal", num_frames, self.n_h*self.n_w, transformer_type, dropout))

        
        self.layer_norm1 = nn.LayerNorm(d_model)
        if self.autoencoder_type == "linear":
            self.embedding = LinearPatchEncoder(d_model, patch_size, image_size, num_frames)
            self.pos_encoding = PositionalEncoding(num_frames, (image_size // patch_size)**2, d_model)
            self.linear_decoder = LinearPatchDecoder(d_model, patch_size, image_size)
        elif self.autoencoder_type == "unet":
            self.unet = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
                encoder_depth = 4,              # output of resnet34 at 4th level is the same as linear autoencoder (B, T, 8*8, 256)
                decoder_channels = (128, 64, 32, 16)
            )
    
        # load nowcast model and it's transform for inference
        self.nowcast_model_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.nowcast_model = torch.jit.load("/home/jovyan/arquivos/solar_forecasting/projeto_visao_comp/model_checkpoints/rn50_nowcast.pt")
        # freeze nowcast model layers
        for param in self.nowcast_model.parameters():
            param.requires_grad = False
            
        self.criterion = nn.MSELoss()

        metrics_img = {f"ssim{i}": torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0) for i in range(5)}
        metrics_img = MetricCollection(metrics_img)
        self.val_metrics_img = metrics_img.clone(prefix="val_")
        self.test_metrics_img = metrics_img.clone(prefix="test_")
        self.val_metrics_img.persistent(True)
        self.test_metrics_img.persistent(True)
        
        metrics_ghi = {f"rmse{i}": torchmetrics.regression.MeanSquaredError(squared=False) for i in range(5)}
        metrics_ghi = MetricCollection(metrics_ghi)
        self.val_metrics_ghi = metrics_ghi.clone(prefix="val_")
        self.test_metrics_ghi = metrics_ghi.clone(prefix="test_")
        self.val_metrics_ghi.persistent(True)
        self.test_metrics_ghi.persistent(True)
        
    # log training time
    def on_train_start(self):
        self.total_start_time = time.time()
    def on_train_end(self):
        self.total_end_time = time.time()
        train_total_time = (self.total_end_time - self.total_start_time) / 60 
        self.logger.experiment.log_metric("train_total_time", train_total_time) 
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    def on_train_epoch_end(self):
        self.epoch_end_time = time.time()
        train_epoch_time = (self.epoch_end_time - self.epoch_start_time) / 60
        self.log("train_epoch_time", train_epoch_time)

    def encode_patches(self, x):
        # x is image sequences (B, T, C, H, W)
        if self.autoencoder_type == "linear":
            x = self.embedding(x) # here x is a tensor of shape (B, T, N_p, d_model)
            x = self.pos_encoding(x)
            return x
        elif self.autoencoder_type == "unet":
            B, T, C, H, W = x.shape
            x = x.flatten(0, 1) #flatten batch and time dimension before unet
            x = self.unet.encoder(x) # here x is a list of tensors for every unet level. The last tensor has shape (B*T, d_model, n_h, n_w), which is reshaped to (B, T, N_p, d_model)
            skip_con_features = x # x is a list of tensors for every unet level
            x = skip_con_features[-1] # (B*T, d_model, n_h, n_w)
            x = x.reshape(B, T, self.d_model, self.n_h*self.n_w)
            x = x.permute(0, 1, 3, 2)
            return x, skip_con_features

    def decode_patches(self, x):
        if self.autoencoder_type == "linear":
            x = self.linear_decoder(x)
        elif self.autoencoder_type == "unet":
            x = self.unet.decoder(x)
            x = self.unet.segmentation_head(x)
        return x

    def transform_patches(self, x):
        B, T, N_p, d_model = x.shape
        if self.attn_type == "space_time":
            x = x.reshape(B, T*N_p, d_model)
            for encoder_i in self.encoder_stack:
                x = encoder_i(x) 
            x = x.reshape(B, T, N_p, d_model)
        elif self.attn_type == "quad_TSST":
            for i in range(0, self.N, 4):
                # temporal attention 1
                x = x.permute(0, 2, 1, 3) # (B, T, N_p, d_model) -> (B, N_p, T, d_model)
                x = x.reshape(B*N_p, T, d_model)
                x = self.encoder_stack[i](x) 

                # spatial attention 1
                x = x.reshape(B, N_p, T, d_model)
                x = x.permute(0, 2, 1, 3) # (B, N_p, T, d_model) -> (B, T, N_p, d_model)
                x = x.reshape(B*T, N_p, d_model)
                x = self.encoder_stack[i+1](x) 

                # spatial attention 2
                x = self.encoder_stack[i+2](x) 

                # temporal attention 2
                x = x.reshape(B, T, N_p, d_model)
                x = x.permute(0, 2, 1, 3) # (B, T, N_p, d_model) -> (B, N_p, T, d_model)
                x = x.reshape(B*N_p, T, d_model)
                x = self.encoder_stack[i+3](x) 
    
                x = x.reshape(B, N_p, T, d_model)
                x = x.permute(0, 2, 1, 3) # (B, N_p, T, d_model) -> (B, T, N_p, d_model)
        return self.layer_norm1(x)    
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.encode_patches(x)
        if self.autoencoder_type == "unet":
            x, skip_con_features = x 
            
        x = self.transform_patches(x)

        if self.autoencoder_type == "linear":
            x = self.decode_patches(x)
        elif self.autoencoder_type == "unet":
            x = x.permute(0, 1, 3, 2) # (B, T, N_p, d_model) -> (B, T, d_model, N_p)
            x = x.reshape(B*T, self.d_model, self.n_h, self.n_w)
            skip_con_features[-1] = x # update unet bottleneck embedding based on predformer prediction
            x = skip_con_features
            x = self.decode_patches(x)
            x = x.reshape(B, T, C, H, W)
        return x
        
    def training_step(self, batch, batch_idx):
        src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis = batch
        if self.transformer_type == "encoder_only":
            x, y = src_imgs, tgt_imgs #B, T, C, H, W
        elif self.transformer_type == "decoder_only":
            #concatenate along time axis
            all_imgs = torch.cat((src_imgs, tgt_imgs), dim=1)
            #shift inputs and outputs by one frame (i.e. N tokens)
            x = all_imgs[:, :-1]
            y = all_imgs[:, 1:]

        #forward
        preds = self(x)
        loss = self.criterion(preds, y)  

        if self.trainer.logger:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0])
        return loss
        
    def sample(self, batch, batch_idx):
        """
        Get prediciton for a single video sample, useful to get predictions outside of training routine
        """
        if batch[0].device != self.device:
            batch = [tensor_.to(self.device) for tensor_ in batch]
        src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis = batch
        if self.transformer_type == "encoder_only":
            x, y = src_imgs, tgt_imgs #B, T, C, H, W
        elif self.transformer_type == "decoder_only":
            all_imgs = torch.cat((src_imgs, tgt_imgs), dim=1)
            y = all_imgs[:, 1:]
        with torch.no_grad():
            if self.transformer_type == "encoder_only":
                inputs_ = x
                preds = self(x)
            elif self.transformer_type == "decoder_only":
                #during inference, start with tensor of zeros for predictions
                preds = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
                #the first frames will always be known during inference and will be equal to src_imgs (t-8, ..., t) 
                preds[:, :src_imgs.shape[1]] = src_imgs
                #shift the preds tensor by one frame to the right
                preds = preds[:, 1:]

                #same as preds tensor, except it will not be shifted since this will be the inputs to the transformer
                inputs_ = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
                inputs_[:, :src_imgs.shape[1]] = src_imgs
                inputs_ = inputs_[:, :-1]

                for t in range(5):
                    x = self(inputs_)
                    preds[:, t+4, :, :, :] = x[:, t+4, :, :, :] 
                    if t < 4:
                        inputs_[:, t+5, :, :, :] = x[:, t+4, :, :, :]
                preds = preds[:, 4:]
                inputs_ = inputs_[:, :5]
                y = y[:, 4:]
            ghi_preds = self.get_ghi_preds(preds, batch)

        return {
            "inputs": inputs_.cpu(),
            "preds": preds.cpu(),
            "trues": y.cpu(),
            "ghi_preds": ghi_preds.cpu(), 
            "src_ghi": src_ghi.cpu(),
            "tgt_ghi": tgt_ghi.cpu(),
            "src_ghi_solis": src_ghi_solis.cpu(),
            "tgt_ghi_solis": tgt_ghi_solis.cpu(),
            "src_kt_solis": src_kt_solis.cpu(),
            "tgt_kt_solis": tgt_kt_solis.cpu()
        }
            
    def validation_step(self, batch, batch_idx):
        src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis = batch
        
        if self.transformer_type == "encoder_only":
            x, y = src_imgs, tgt_imgs #B, T, C, H, W
            inputs_ = x
            preds = self(x)
        elif self.transformer_type == "decoder_only":
            all_imgs = torch.cat((src_imgs, tgt_imgs), dim=1)
            y = all_imgs[:, 1:]
            #during inference, start with tensor of zeros for predictions
            preds = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
            #the first frames will always be known during inference and will be equal to src_imgs (t-8, ..., t) 
            preds[:, :src_imgs.shape[1]] = src_imgs
            #shift the preds tensor by one frame to the right
            preds = preds[:, 1:]

            #same as preds tensor, except it will not be shifted since this will be the inputs to the transformer
            inputs_ = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
            inputs_[:, :src_imgs.shape[1]] = src_imgs
            inputs_ = inputs_[:, :-1]

            for t in range(5):
                x = self(inputs_)

                preds[:, t+4, :, :, :] = x[:, t+4, :, :, :] 
                if t < 4:
                    inputs_[:, t+5, :, :, :] = x[:, t+4, :, :, :]  

            preds = preds[:, 4:]
            inputs_ = inputs_[:, :5]
            y = y[:, 4:]

        if self.trainer.logger:
            #log video mse
            val_loss = self.criterion(preds, y)
            self.log("val_loss", val_loss, on_epoch=True, logger=True)

            #log framewise ssim
            for t, ssim_ in enumerate(self.val_metrics_img):
                self.val_metrics_img[ssim_].update(preds[:, t], y[:, t])
            self.log_dict(self.val_metrics_img, on_epoch=True, logger=True)

            #log framewise ghi
            ghi_preds = self.get_ghi_preds(preds, batch)
            for t, rmse_ in enumerate(self.val_metrics_ghi):
                self.val_metrics_ghi[rmse_].update(ghi_preds[:, t], tgt_ghi[:, t])
            self.log_dict(self.val_metrics_ghi, on_epoch=True, logger=True)
        
            
            
    def test_step(self, batch, batch_idx):
        src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis = batch
        
        if self.transformer_type == "encoder_only":
            x, y = src_imgs, tgt_imgs #B, T, C, H, W
            inputs_ = x
            preds = self(x)
        elif self.transformer_type == "decoder_only":
            all_imgs = torch.cat((src_imgs, tgt_imgs), dim=1)
            y = all_imgs[:, 1:]
            #during inference, start with tensor of zeros for predictions
            preds = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
            #the first frames will always be known during inference and will be equal to src_imgs (t-8, ..., t) 
            preds[:, :src_imgs.shape[1]] = src_imgs
            #shift the preds tensor by one frame to the right
            preds = preds[:, 1:]

            #same as preds tensor, except it will not be shifted since this will be the inputs to the transformer
            inputs_ = torch.zeros_like(torch.cat((src_imgs, tgt_imgs), dim=1))
            inputs_[:, :src_imgs.shape[1]] = src_imgs
            inputs_ = inputs_[:, :-1]

            for t in range(5):
                x = self(inputs_)

                preds[:, t+4, :, :, :] = x[:, t+4, :, :, :] 
                if t < 4:
                    inputs_[:, t+5, :, :, :] = x[:, t+4, :, :, :]

            preds = preds[:, 4:]
            inputs_ = inputs_[:, :5]
            y = y[:, 4:]

        if self.trainer.logger:
            #log video mse
            test_loss = self.criterion(preds, y)
            self.log("test_loss", test_loss, on_epoch=True, logger=True)

            #log framewise ssim
            for t, ssim_ in enumerate(self.test_metrics_img):
                self.test_metrics_img[ssim_].update(preds[:, t], y[:, t])
            self.log_dict(self.test_metrics_img, on_epoch=True, logger=True)

            #log framewise ghi
            ghi_preds = self.get_ghi_preds(preds, batch)
            for t, rmse_ in enumerate(self.test_metrics_ghi):
                self.test_metrics_ghi[rmse_].update(ghi_preds[:, t], tgt_ghi[:, t])
            self.log_dict(self.test_metrics_ghi, on_epoch=True, logger=True)

    def get_ghi_preds(self, img_preds, batch):
        src_imgs, tgt_imgs, src_ghi, tgt_ghi, src_ghi_solis, tgt_ghi_solis, src_kt_solis, tgt_kt_solis = batch
        B, T, C, H, W = img_preds.shape
        img_preds = img_preds.reshape(B*T, C, H, W)
        img_preds = self.nowcast_model_transform(img_preds)
        with torch.no_grad():
            kt_preds = self.nowcast_model(img_preds).squeeze()
        kt_preds = kt_preds.reshape(B, T)
        ghi_preds = kt_preds * tgt_ghi_solis
        return ghi_preds
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2
        )
        if self.lr_scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.learning_rate, total_steps=self.sched_max_steps, final_div_factor=1e4
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
               }
            }
