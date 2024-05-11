from abstractjepa import JEPA
import pytorch_lightning as pl

class ChessJEPA(JEPA, pl.LightningModule):
    def __init__(self, context_encoder, predictor, loss_calculator, board_state_embedder):
        super(ChessJEPA, self).__init__(context_encoder, predictor, loss_calculator)
        self.board_state_embedder = board_state_embedder
        self.learning_rate = 0.0001 # placeholder

    def encode_x(self, x):
        pass

    def encode_y(self, y):
        pass

    def predict_encoded_y(self, encoded_x, z):
        pass

    def get_loss(self, encoded_y, encoded_yhat, z=None):
        pass

    def forward(self, x, z):
        embedded_states = self.board_state_embedder(x)
        return embedded_states

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        encoded_x = self(x)
        pass

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        encoded_x = self(x)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

