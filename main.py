from ChessDataModule import ChessDataModule
from ChessJEPA import ChessJEPA
from PieceEmbedding import PieceEmbedding
from constants import *
import torch.nn as nn

def run_passthrough_job():
    dataModule = ChessDataModule('data/all_with_filtered_anotations_since1998.txt', maxGames=100)
    dataModule.prepare_data()
    dataModule.setup(None)
    d_model = 128
    action_size = 4
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    predictor = nn.RNNCell(input_size=action_size, hidden_size=32*d_model)
    model = ChessJEPA(transformer_encoder, predictor, None, PieceEmbedding(d_model=d_model))
    for batch_idx, batch in enumerate(dataModule.train_dataloader()):
        x, y, z = batch
        print(batch_idx)
        print(z.shape)
        print(x.shape)
        output = model(x, z)
        encoded_x, predicted_encoded_y = output
        print(encoded_x.shape)
        print(predicted_encoded_y.shape)
        print()
    print('complete')

if __name__ == "__main__":
    run_passthrough_job()

