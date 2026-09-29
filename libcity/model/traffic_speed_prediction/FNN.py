from logging import getLogger

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class FNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))

        self.hidden_size = config.get('hidden_size', 128)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.fc1 = nn.Linear(self.input_window * self.feature_dim, self.hidden_size)
        self.relu = nn.ReLU()
        self.hidden_size_2 = config.get('hidden_size_2')
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size_2 or
                             self.output_window * self.output_dim)
        self.fc3 = (nn.Linear(self.hidden_size_2, self.output_window * self.output_dim)
                    if self.hidden_size_2 else None)
        self.residual_anchor = config.get('residual_anchor')
        self.residual_scale = config.get('residual_scale', 1.0)
        if self.residual_anchor not in (None, 'mean', 'last'):
            raise ValueError('residual_anchor must be mean, last or None')

    def forward(self, batch):
        observations = batch['X']
        inputs = observations
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0, 2, 1, 3)
        inputs = inputs.reshape(batch_size, self.num_nodes, -1)
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        if self.fc3 is not None:
            outputs = self.fc3(self.relu(outputs))
        outputs = outputs.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)
        if self.residual_anchor is not None:
            history = observations[..., :self.output_dim]
            anchor = (history.mean(dim=1) if self.residual_anchor == 'mean'
                      else history[:, -1])
            outputs = anchor.unsqueeze(2) + self.residual_scale * outputs
        return outputs.permute(0, 2, 1, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
