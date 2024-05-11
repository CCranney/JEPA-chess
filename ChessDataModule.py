import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
import chess
import chess.pgn
import pandas as pd
from constants import ALL_PIECES

class ChessDataModule(pl.LightningDataModule):
    def __init__(self, filePath, maxGames = 5000):
        super().__init__()
        self.filePath = filePath
        self.maxGames = maxGames

    def prepare_data(self):
        list_of_moves_per_game = load_game_moves_from_kaggle_dataset(self.filePath, self.maxGames)
        self.X, self.Y, self.Z = process_moves_into_board_state_transitions(list_of_moves_per_game)

    def setup(self, stage):
        val_ratio = 0.2
        num_training_samples = int((1 - val_ratio) * len(self.X))
        indices = torch.arange(len(self.X)).long()
        train_indices = indices[:num_training_samples].detach().cpu().numpy()
        val_indices = indices[num_training_samples:].detach().cpu().numpy()

        # Separate features and targets for both datasets
        self.train_dataset = TensorDataset(self.X[train_indices], self.Y[train_indices], self.Z[train_indices])
        self.val_dataset = TensorDataset(self.X[val_indices], self.Y[val_indices], self.Z[val_indices])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)

def load_game_moves_from_kaggle_dataset(filePath, maxGames):
    with open(filePath, 'r') as f:
        lines = [line.strip() for line in f.readlines()[:maxGames] if not line.startswith('#')]
    move_lists = []
    for line in lines:
        all_values = line.split()
        moves = [move.split('.', 1)[1] for move in all_values[17:]]
        move_lists.append(moves)
    return move_lists

def process_board_state(strings):
    piece_indices = [i for i, s in enumerate(strings) if s != '.']
    piece_coordinates = [(i // 8, i % 8) for i in piece_indices]
    numeric_pieces = {s: i for i, s in enumerate(ALL_PIECES)}
    numeric_representation = torch.full((3,32), fill_value=0, dtype=torch.int32) # the first "piece" token is for captured pieces (X). These will be masked.
    for i, (letter, number) in enumerate(piece_coordinates):
        insert = torch.tensor([numeric_pieces[strings[piece_indices[i]]], letter, number], dtype=torch.int32)
        numeric_representation[:, i].scatter_(dim=0, index=torch.arange(3), src=insert)
    return numeric_representation

def process_moves_into_board_state_transitions(list_of_moves_per_game):
    X, Y, Z = [], [], []
    for i in range(len(list_of_moves_per_game)):
        board = chess.Board()
        try:
            for move in list_of_moves_per_game[i]:
                move_obj = board.parse_san(move)
                Z.append(convert_move_to_tensor(move_obj))
                premoveState = convert_board_to_state(board)
                X.append(process_board_state(premoveState))
                board.push(move_obj)
                postmoveState = convert_board_to_state(board)
                Y.append(process_board_state(postmoveState))
        except:
            continue
    return torch.stack(X), torch.stack(Y), torch.stack(Z)

def convert_board_to_state(board):
    return list(
        "".join(["." * int(char) if char.isdigit() else char for char in board.fen().replace("/", "")])[:64]
    )

def convert_move_to_tensor(move):
    letters = 'abcdefgh'
    coordinates = list(str(move))
    return torch.tensor([letters.index(coordinates[0]), int(coordinates[1])-1, letters.index(coordinates[2]), int(coordinates[3])-1], dtype=torch.int32)


