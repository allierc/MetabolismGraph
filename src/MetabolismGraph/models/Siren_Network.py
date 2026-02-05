# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, learnable_omega=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features
        self.learnable_omega = learnable_omega

        if learnable_omega:
            self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        else:
            self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights(omega_0)

    def init_weights(self, omega_0):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega_0,
                                             np.sqrt(6 / self.in_features) / omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., learnable_omega=False):
        super().__init__()

        self.learnable_omega = learnable_omega
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0, learnable_omega=learnable_omega))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, learnable_omega=learnable_omega))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0, learnable_omega=learnable_omega))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    def get_omegas(self):
        """return current omega values for monitoring (only useful when learnable_omega=True)."""
        omegas = []
        for layer in self.net:
            if hasattr(layer, 'omega_0') and hasattr(layer, 'learnable_omega'):
                if isinstance(layer.omega_0, nn.Parameter):
                    omegas.append(layer.omega_0.item())
                else:
                    omegas.append(layer.omega_0)
        return omegas

    def get_omega_L2_loss(self):
        """return L2 regularization loss on omega parameters (encourages smaller omega)."""
        loss = 0.0
        for layer in self.net:
            if hasattr(layer, 'omega_0') and hasattr(layer, 'learnable_omega'):
                if layer.learnable_omega and isinstance(layer.omega_0, nn.Parameter):
                    loss = loss + layer.omega_0 ** 2
        return loss


class small_Siren(nn.Module):
    def __init__(self, in_features=1, hidden_features=128, hidden_layers=3, out_features=1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        layers.append(nn.Linear(hidden_features, out_features))  # final linear layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Siren_Network(nn.Module):
    def __init__(self, image_width, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., device='cuda:0'):
        super().__init__()

        self.device = device
        self.image_width = image_width

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

        self.net = self.net.to(device)

        self.coords = None

    @property
    def values(self):
        # Call forward method
        output, coords = self.__call__()
        return output.squeeze().reshape(self.image_width, self.image_width)

    def coordinate_grid(self, n_points):
        coords = np.linspace(0, 1, n_points, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        return xy_grid

    def get_mgrid(self, sidelen, dim=2, enlarge=False):
        '''generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
        if enlarge:
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen*20)])
        else:
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])

        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def forward(self, coords=None, time=None, enlarge=False):

        if coords is None:
            coords = self.get_mgrid(self.image_width, dim=2, enlarge=enlarge).to(self.device)
            if time is not None:
               coords = torch.cat((coords, time * torch.ones_like(coords[:, 0:1])), 1)

        output = self.net(coords)
        return output
