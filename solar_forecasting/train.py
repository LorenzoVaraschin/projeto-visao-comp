import comet_ml
import lightning as L
import os
from lightning.pytorch.loggers import CometLogger
from argparse import ArgumentParser
import data
from predformer import PredFormer
from lightning.pytorch.callbacks import LearningRateMonitor
import os
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--h", help="Num de heads do transformer", type=int, default=8)
    parser.add_argument("--d_model", help="Dim dos patch embeddings", type=int, default=256)
    parser.add_argument("--d_hidden", help="Dim da camada oculta do bloco GLU", type=int, default=1024)
    parser.add_argument("--N", help="Num de blocos do transformer. OBS: No relatorio foi usada a letra L para essa variavel", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_frames", help="Num de frames previstos. T=5 para encoder only e T=9 para decoder only", type=int, default=9)
    parser.add_argument("--attn_type", type=str, default="space_time")
    parser.add_argument("--lr_scheduler", type=str, default="one_cycle")
    parser.add_argument("--transformer_type", type=str, default="decoder_only")
    parser.add_argument("--autoencoder_type", type=str, default="linear")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--log_run", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    return args
    
def main(args):
    if args.transformer_type == "decoder_only":
        assert args.num_frames == 9, f"Num frames must be 9 in decoder only transformer, got {args.num_frames}"
    elif args.transformer_type == "encoder_only":
        assert args.num_frames == 5, f"Num frames must be 5 in encoder only transformer, got {args.num_frames}"
    train_dataloader, val_dataloader, test_dataloader = data.make_dataloaders(args.batch_size)
    image_size = next(iter(train_dataloader))[0].shape[-1]
    assert image_size == args.image_size
    callbacks_ = list()
    model = PredFormer(
        h = args.h,
        d_model = args.d_model,
        d_hidden = args.d_hidden,
        N = args.N,
        patch_size = args.patch_size,
        image_size = args.image_size,
        dropout = args.dropout,
        num_frames = args.num_frames,
        attn_type = args.attn_type,
        transformer_type = args.transformer_type,
        autoencoder_type = args.autoencoder_type,
        sched_max_steps = len(train_dataloader) * args.max_epochs,
        learning_rate = args.learning_rate,
        lr_scheduler = args.lr_scheduler
    )
    exp_name = f"{args.transformer_type}_{args.autoencoder_type}_{args.attn_type}"
    if args.log_run:
        model_logger = CometLogger(
            api_key="XXXXXXXXXXXXXXXXXXXXXXXX",
            project_name="xxxxx-xxxxx-xxxxxx",
            experiment_name=exp_name
        )
        callbacks_.append(LearningRateMonitor(logging_interval='epoch'))
    else:
        model_logger = False
        
    callbacks_.append(ModelCheckpoint(dirpath=os.path.join(os.getcwd(), "model_checkpoints"), filename=exp_name))

    trainer = L.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=model_logger,
        callbacks=callbacks_,
        accumulate_grad_batches=args.accumulate_grad_batches,
        devices=[args.device]
    )
    
    if args.log_run:
        trainer.logger.experiment.log_code("predformer.py")
        trainer.logger.experiment.log_code("aux_modules.py")
        trainer.logger.experiment.log_code("data.py")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    seed_ = 0
    L.seed_everything(seed_)
    main(args)
