from typing import Callable, Tuple

import torch
import torch.nn as nn

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """


class Embedder:
    def __init__(self, **kwargs) -> None:
        super.__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires: int, input_dims: int = 3) -> Tuple[Callable, int]:
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


class PositionalEncoding(nn.Module):
    def __init__(self, level: int = 10) -> None:
        super().__init__()
        self.level = level

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        pi = 1.0
        p_transformed = torch.cat(
            [
                torch.cat([torch.sin((2**i) * pi * p), torch.cos((2**i) * pi * p)], dim=-1)
                for i in range(self.level)
            ],
            dim=-1,
        )

        return torch.cat([p, p_transformed], dim=-1)

    def __repr__(self) -> str:
        return f"PositionalEncoding(level={self.level})"
