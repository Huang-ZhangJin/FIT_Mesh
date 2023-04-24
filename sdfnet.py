import os
from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

from embedder import PositionalEncoding


class SDFNetwork(nn.Module):
    def __init__(
        self,
        dim_output: int,
        dim_hidden: int = 256,
        n_layers: int = 8,
        skip_in: Sequence[int] = (4,),
        level: int = 10,
        geometric_init: bool = True,
        weight_norm: bool = True,
        inside_outside: bool = False,
        bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.frequency_encoder = PositionalEncoding(level)

        input_ch = level * 6 + 3

        dims = [input_ch] + [dim_hidden for _ in range(n_layers)] + [dim_output]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif level > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif level > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if l < self.num_layers - 2:
                setattr(self, "act" + str(l), nn.Softplus(beta=100))
                # setattr(self, "act" + str(l), nn.ReLU(inplace=True))

    def forward(self, inputs: torch.Tensor, with_normals: bool = True) -> torch.Tensor:
        if with_normals:
            inputs.requires_grad_(True)

        x = self.frequency_encoder(inputs)
        x_enc = x
        for l in range(0, self.num_layers - 1):
            lin: nn.Linear = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, x_enc], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = getattr(self, "act" + str(l))(x)

        if not with_normals:
            return x
        else:
            sdf = x[:, :1]
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            normals = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return x, normals

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, with_normals=False)[:, :1]

