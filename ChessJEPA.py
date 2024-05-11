from abstractjepa import JEPA
import pytorch_lightning as pl
import torch

class ChessJEPA(JEPA, pl.LightningModule):
    def __init__(self, context_encoder, predictor, loss_calculator, board_state_embedder):
        super(ChessJEPA, self).__init__(context_encoder, predictor, loss_calculator)
        self.board_state_embedder = board_state_embedder
        self.learning_rate = 0.0001 # placeholder

    def encode_x(self, x):
        return self.context_encoder(x)

    def encode_y(self, y):
        return self.target_encoder(y)

    def predict_encoded_y(self, encoded_x, z):
        encoded_x_flat = encoded_x.view(-1, encoded_x.shape[-2] * encoded_x.shape[-1])
        predicted_encoded_y_flat = self.predictor(z, encoded_x_flat)
        predicted_encoded_y = predicted_encoded_y_flat.view(-1, encoded_x.shape[-2], encoded_x.shape[-1])
        return predicted_encoded_y

    def get_loss(self, encoded_x, encoded_y, predicted_encoded_y):
        variance_loss, covariance_loss, invariance_loss = self.loss_calculators['vicreg'].calculate_VICReg_loss(encoded_x, encoded_y, predicted_encoded_y)
        return variance_loss + covariance_loss + invariance_loss

    def forward(self, x, z):
        embedded_x = self.board_state_embedder(x)
        encoded_x = self.encode_x(embedded_x)
        predicted_encoded_y = self.predict_encoded_y(encoded_x, z)
        return encoded_x, predicted_encoded_y

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        encoded_x, predicted_encoded_y = self(x, z)
        encoded_y = self.encode_y(self.board_state_embedder(y))
        loss = self.get_loss(encoded_x, encoded_y, predicted_encoded_y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        encoded_x, predicted_encoded_y = self(x, z)
        encoded_y = self.encode_y(self.board_state_embedder(y))
        loss = self.get_loss(encoded_x, encoded_y, predicted_encoded_y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

