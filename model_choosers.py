import torch.nn as nn
from models.embedders.PieceEmbedding import PieceEmbedding
from models.loss_calculators.VicRegLossCalculator import VICRegLossCalculator

'''
# NOTE: This is an initialization of the file. In the future, as more encoders, predictors etc. are developed, 
    the idea is to allow the user to choose between options that are plugged in accordingly.
'''

def choose_models(args):
    return choose_encoder(args), choose_predictor(args), choose_loss_calculators(args), choose_embedder(args)

def choose_embedder(args):
    return PieceEmbedding(d_model=args['d_model'])

def choose_encoder(args):
    encoder_layer = nn.TransformerEncoderLayer(d_model=args['d_model'], nhead=8, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    return transformer_encoder

def choose_predictor(args):
    return nn.RNNCell(input_size=args['action_size'], hidden_size=32*args['d_model'])

def choose_loss_calculators(args):
    expander = choose_expander(args)
    return {'vicreg': VICRegLossCalculator(expander)}

def choose_expander(args):
    return nn.Linear(args['num_pieces'] * args['d_model'], args['num_pieces'] * args['d_model'] * args['expansion_scale'])
