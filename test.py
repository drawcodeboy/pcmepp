"""test code for PCME++
"""
from gc import callbacks
import os
import fire

import torch
from transformers import BertTokenizer

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from config import parse_config
from logger import PCMEPPLogger, PCMEPPLogger_Wandb

from pcmepp.datasets import get_loaders, get_test_loader
from pcmepp.engine import PCMEPPModel


def main(config_path, load_from_checkpoint=None, **kwargs):
    # Load configuration
    if load_from_checkpoint:
        print(f'Resume from the previous weight {load_from_checkpoint=}')
        ckpt = torch.load(load_from_checkpoint)
        config = ckpt['hyper_parameters']['opt']
        for arg_key, arg_val in kwargs.items():
            keys = arg_key.split('__')
            n_keys = len(keys)

            _config = config
            for idx, _key in enumerate(keys):
                if n_keys - 1 == idx:
                    _config[_key] = arg_val
                else:
                    _config = _config[_key]
    else:
        config = parse_config(config_path,
                              strict_cast=False,
                              **kwargs)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab

    # Data loader
    train_loader, val_loader = get_loaders(
        **config.dataloader, tokenizer=tokenizer, opt=config, vocab_size=len(vocab))
    te_loader = get_test_loader('testall', 'coco', tokenizer,
                                config.dataloader.eval_batch_size, config.dataloader.workers, config, len(vocab))

    # Define model
    model = PCMEPPModel(config)

    if config.train.get('strategy'):
        strategy = config.train.strategy
    else:
        strategy = DDPStrategy(
            # No way to avoid find_unused_parameters=True. https://github.com/pytorch/pytorch/issues/22049#issuecomment-505617666
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    trainer = pl.Trainer(
        strategy=strategy,
        callbacks=callbacks,
        precision=config.train.precision,
        gradient_clip_val=config.train.grad_clip,
        log_every_n_steps=config.train.log_step,
        max_epochs=config.train.train_epochs,
        num_nodes=int(config.train.get('world_size', 1)),
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        benchmark=True,
        num_sanity_val_steps=0,
    )
    # Only evaluating
    # trainer.fit(model, train_dataloaders=train_loader,
    #             val_dataloaders=val_loader, ckpt_path=load_from_checkpoint)

    # if model.swa_enabled:
    #     model.print('evaluate by SWA')
    #     model.eval_by_swa = True
    #     trainer.validate(model, val_loader)
    trainer.validate(model, [val_loader, te_loader], ckpt_path=load_from_checkpoint)

if __name__ == '__main__':
    fire.Fire(main)
