import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        output = x

        with torch.no_grad():
            for layer in model:
                output = layer(output)

                if isinstance(layer, nn.Linear):
                    dead = (output <= 0).all(dim=0).float().mean().item()

                    stats.append({
                        "mean": round(output.mean().item(), 4),
                        "std": round(output.std().item(), 4),
                        "dead_fraction": round(dead, 4)
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"

        if any(s["norm"] > 1000 for s in gradient_stats):
            return "exploding_gradients"

        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        if any(s["std"] < 0.1 for s in activation_stats):
            return "vanishing_gradients"

        if any(s["std"] > 10.0 for s in activation_stats):
            return "exploding_gradients"

        return "healthy"