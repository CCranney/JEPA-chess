from ChessDataModule import ChessDataModule
from ChessJEPA import ChessJEPA
from PieceEmbedding import PieceEmbedding
from constants import *

def run_passthrough_job():
    dataModule = ChessDataModule('data/all_with_filtered_anotations_since1998.txt', maxGames=100)
    dataModule.prepare_data()
    dataModule.setup(None)
    model = ChessJEPA(None, None, None, PieceEmbedding(d_model=128))
    for batch_idx, batch in enumerate(dataModule.train_dataloader()):
        x, y, z = batch
        print(batch_idx)
        print(x.shape)
        output = model(x, z)
        print(output.shape)
        print()
    print('complete')

if __name__ == "__main__":
    run_passthrough_job()

