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

def process_moves_into_board_state_transitions(list_of_moves_per_game):
    X, Y, Z = [], [], []
    for i in range(len(list_of_moves_per_game)):
        try:
            board = chess.Board()
            piece_tracker = create_piece_tracker(board)
            for move in list_of_moves_per_game[i]:
                move_obj = board.parse_san(move)
                Z.append(convert_move_to_tensor(move_obj))
                assert sorted(piece_tracker) == sorted(create_piece_tracker(board))
                X.append(process_pieces_for_model(piece_tracker))
                board.push(move_obj)
                piece_tracker = update_piece_tracker(piece_tracker, board, move_obj)
                Y.append(process_pieces_for_model(piece_tracker))
        except:
            continue
    return torch.stack(X), torch.stack(Y), torch.stack(Z)

def convert_move_to_tensor(move):
    letters = 'abcdefgh'
    coordinates = list(str(move))
    return torch.tensor([letters.index(coordinates[0]), int(coordinates[1])-1, letters.index(coordinates[2]), int(coordinates[3])-1], dtype=torch.float32)

def convert_board_to_state(board):
    return list(
        "".join(["." * int(char) if char.isdigit() else char for char in board.fen().replace("/", "")])[:64]
    )

def process_pieces_for_model(piece_tracker):
    numeric_pieces = {s: i for i, s in enumerate(ALL_PIECES)}
    piece_values = [(numeric_pieces[piece], position // 8, position % 8) for piece, position in piece_tracker]
    return_value = torch.tensor(piece_values, dtype=torch.int32)
    return return_value

def create_piece_tracker(board, final_length=32):
    piece_tracker = []
    boardState = convert_board_to_state(board)
    for i, piece in enumerate(boardState):
        if piece=='.': continue
        piece_tracker.append((piece, i))
    for i in range(32-len(piece_tracker)):
        piece_tracker.append(('X',0)) # indicates captured pieces
    return piece_tracker

def update_piece_tracker(piece_tracker, board, move):
    temp_piece_tracker = create_piece_tracker(board)
    changed_indices = [i for i, piece_and_position in enumerate(piece_tracker) if piece_and_position not in temp_piece_tracker]
    new_piece_states = set(temp_piece_tracker) - set(piece_tracker)
    new_piece_states_dict = {piece_and_position[0]:piece_and_position for piece_and_position in new_piece_states}
    for i in changed_indices:
        if piece_tracker[i][0] in new_piece_states_dict:
            piece_tracker[i] = new_piece_states_dict[piece_tracker[i][0]]
        elif str(move)[-1] in ['q','r','n','b'] and piece_tracker[i][0] == 'p': # promoting
            piece_tracker[i] = new_piece_states_dict[str(move)[-1]]
        elif str(move)[-1] in ['q','r','n','b'] and piece_tracker[i][0] == 'P': # promoting
            piece_tracker[i] = new_piece_states_dict[str(move)[-1].upper()]
        else:
            piece_tracker[i] = ('X',0) # captured
    return piece_tracker