from abstractjepa import JEPA


class ChessJEPA(JEPA):
    def __init__(self, context_encoder, predictor, loss_calculator):
        super(ChessJEPA, self).__init__(context_encoder, predictor, loss_calculator)

    def encode_x(self, x):
        pass

    def encode_y(self, y):
        pass

    def predict_encoded_y(self, encoded_x, z):
        pass

    def get_loss(self, encoded_y, encoded_yhat, z=None):
        pass

    def forward(self, states, actions):
        pass
