from ChessDataModule import ChessDataModule
from ChessJEPA import ChessJEPA
from constants import *
import torch.nn as nn
import time
from pytorch_lightning import Trainer
from model_choosers import choose_models

def run_passthrough_job():
    args = {
    'd_model':128,
    'action_size':4,
    'num_pieces':32,
    'expansion_scale':4,
    }
    encoder, predictor, loss_calculators, embedder = choose_models(args)
    model = ChessJEPA(encoder, predictor, loss_calculators, embedder)
    dataModule = ChessDataModule('data/all_with_filtered_anotations_since1998.txt', maxGames=100)
    trainer = Trainer(
        logger=False,
        max_epochs=5,
    )

    start = time.time()
    trainer.fit(model, dataModule)
    num_epochs = trainer.current_epoch
    end = time.time()
    train_time = end - start
    print("Training completed in {} epochs.".format(num_epochs))

if __name__ == "__main__":
    run_passthrough_job()

