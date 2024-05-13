
import torch
from torch.nn import functional as F
from abstractjepa.loss import VICRegLoss

class VICRegLossCalculator(VICRegLoss):

    def calculate_variance(self, normalized_expanded_representation):
        std = torch.sqrt(normalized_expanded_representation.var(dim=0) + 0.0001)
        return torch.mean(F.relu(1 - std))

    def calculate_covariance(self, normalized_expanded_representation):
        batch_size = normalized_expanded_representation.shape[0]
        num_features = normalized_expanded_representation.shape[-1]
        cov = (normalized_expanded_representation.T @ normalized_expanded_representation) / (batch_size - 1)
        return off_diagonal(cov).pow_(2).sum().div(num_features)

    def calculate_invariance(self, expanded_representation_yhat, expanded_representation_y):
        return F.mse_loss(expanded_representation_yhat, expanded_representation_y)

    def get_variance_and_covariance_loss(self, expanded_representation):
        normalized_x = expanded_representation - expanded_representation.mean(dim=0)
        variance_loss = self.calculate_variance(normalized_x)
        covariance_loss = self.calculate_covariance(normalized_x)
        return variance_loss, covariance_loss

    def calculate_VICReg_loss(self, encoded_x, encoded_y, predicted_encoded_y):
        encoded_x_flat = encoded_x.view(-1, encoded_x.shape[-2] * encoded_x.shape[-1])
        encoded_y_flat = encoded_y.view(-1, encoded_x.shape[-2] * encoded_x.shape[-1])
        predicted_encoded_y_flat = predicted_encoded_y.view(-1, encoded_x.shape[-2] * encoded_x.shape[-1])

        expanded_encoded_x = self.expander(encoded_x_flat)
        expanded_encoded_y = self.expander(encoded_y_flat)
        expanded_predicted_encoded_y = self.expander(predicted_encoded_y_flat)

        variance_loss, covariance_loss = self.get_variance_and_covariance_loss(expanded_encoded_x)
        invariance_loss = self.calculate_invariance(expanded_predicted_encoded_y,
                                                        expanded_encoded_y)
        return variance_loss, covariance_loss, invariance_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()