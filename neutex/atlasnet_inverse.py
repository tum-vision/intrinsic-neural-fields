"""
Adapted from
https://github.com/fbxiang/NeuTex/blob/master/models/atlasnet/inverse.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from .network_utils import init_weights


class MappingManifold(nn.Module):
    # def __init__(self, code_size, input_dim, output_dim, hidden_size=128, num_layers=2):
    def __init__(self, input_dim, output_dim, hidden_size=128, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        # self.code_size = code_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        # self.linear1 = nn.Linear(self.input_dim, self.code_size)
        # self.linear2 = nn.Linear(self.code_size, self.hidden_neurons)

        # We only keep linear1 since we dont have a latent vector anymore
        self.linear1 = nn.Linear(self.input_dim, self.hidden_neurons)

        init_weights(self.linear1)
        # init_weights(self.linear2)

        self.linear_list = nn.ModuleList(
            [
                nn.Linear(self.hidden_neurons, self.hidden_neurons)
                for _ in range(self.num_layers)
            ]
        )

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    # def forward(self, x, latent):
    def forward(self, x):
        # x = self.linear1(x) + latent[:, None]
        # x = self.activation(x)
        # x = self.activation(self.linear2(x))
        x = self.activation(self.linear1(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        return self.last_linear(x)


# Transforms 3D points on the manifold into uv-map points
class InverseAtlasnet(nn.Module):
    # def __init__(self, num_primitives, code_size, primitive_type="sphere"):
    def __init__(self, num_primitives, primitive_type="sphere"):
        super().__init__()

        if primitive_type == 'square':
            self.output_dim = 2
        else:
            self.output_dim = 3

        self.encoders = nn.ModuleList(
            # Adding an additional dimension to the output_dim for calculating weights
            [MappingManifold(3, self.output_dim + 1) for _ in range(0, num_primitives)]
            # [MappingManifold(code_size, 3, self.output_dim + 1) for _ in range(0, num_primitives)]
        )

    # def forward(self, latent_vector, points):
    def forward(self, points):
        """
        Args:
            points: :math:`(N,*,3)`
        """
        input_shape = points.shape

        points = points.view(points.shape[0], -1, 3)

        output = [
            encoder(points) for encoder in self.encoders
            # encoder(points, latent_vector) for encoder in self.encoders
        ]  # (N, *, 3)[primitives]
        output = torch.stack(output, dim=-2)  # (N, *, primitives, 3)
        output = output.view(input_shape[:-1] + output.shape[-2:])

        if self.output_dim == 2:
            uv = torch.tanh(output[..., :-1])
        else:
            uv = F.normalize(output[..., :-1], dim=-1)

        # Building weights for combining the different primitives
        weights_logits = output[..., -1]
        weights = torch.softmax(weights_logits, dim=-1)
        return uv, weights, weights_logits
