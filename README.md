# projeto-visao-comp

Projeto da disciplina INE410121 - Visão Computacional da Universidade Federal de Santa Catarina (UFSC), ministrada pelo Prof. Dr. Aldo Von Wangenheim.

# Instalação 
Usando a imagem `Lapix` do VLAB, basta instalar os seguintes pacotes
```bash
$ pip install lightning
````
```bash
$ pip install comet_ml # Opcional mas recomendado, somente para logar no CometML 
````
```bash
$ pip install segmentation-models-pytorch 
````

# Quick Start
O arquivo zip com os checkpoints dos modelos estão nessa pasta do [Google Drive](https://drive.google.com/drive/u/0/folders/1saedI3hfRcQFbUau3NcWbm0HQIZR6sr0).

Os checkpoints disponíveis são:
- `_1__encoder_only__linear__space_time.ckpt`;
- `_2__encoder_only__linear__quad_TSST.ckpt`;
- `_3__encoder_only__unet__space_time.ckpt`;
- `_4__encoder_only__unet__quad_TSST.ckpt`;
- `_5__decoder_only__linear__space_time.ckpt`;
- `_6__decoder_only__linear__quad_TSST.ckpt`;
- `_7__decoder_only__unet__space_time.ckpt`;
- `_8__decoder_only__unet__quad_TSST.ckpt`.

```python
from predformer import PredFormer
from data import make_dataloaders

device = 0 # cpu ou número da GPU

# carregar o modelos e seu checkpoint
model = PredFormer.load_from_checkpoint(
  "path/to/checkpoint",
  map_location=f"cuda:{device}" if isinstance(device, int) else device,
).eval()

# criar os data loaders
train_dataloader, val_dataloader, test_dataloader = make_dataloaders(batch_size=16)
batch = next(iter(test_dataloader))
outs_ = model.sample(batch, 0)
```

