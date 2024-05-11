from ChessDataModule import ChessDataModule
from ChessJEPA import ChessJEPA
from PieceEmbedding import PieceEmbedding
from constants import *
import torch.nn as nn
import time
from lossCalculators import VICRegLossDotCalculator
from pytorch_lightning import Trainer


def run_passthrough_job():
    dataModule = ChessDataModule('data/all_with_filtered_anotations_since1998.txt', maxGames=100)
    d_model = 128
    action_size = 4
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    predictor = nn.RNNCell(input_size=action_size, hidden_size=32*d_model)
    expander = nn.Identity()
    loss_calculators = {'vicreg': VICRegLossDotCalculator(expander)}
    model = ChessJEPA(transformer_encoder, predictor, loss_calculators, PieceEmbedding(d_model=d_model))
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

