import torch
from torch import nn

from .state import StateCodec


class RunningMoments(nn.Module):
    """Streaming moments updated once per real training transition, never on replay."""

    def __init__(self, dim, floor=1e-2):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("floor", torch.broadcast_to(torch.as_tensor(floor), (dim,)).clone())

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        if not x.shape[0]:
            return
        n = x.shape[0]
        total = self.count + n
        rate = n / total
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        diff = mean - self.mean
        self.var.copy_((1 - rate) * self.var + rate * var + rate * (1 - rate) * diff.square())
        self.mean.add_(rate * diff)
        self.count.copy_(total)

    def std(self):
        return (self.var + self.floor.square()).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class WorldModel(nn.Module):
    def __init__(self, descriptor, hidden=(2048, 2048, 2048), amp=True):
        super().__init__()
        self.d = descriptor
        self.codec = StateCodec(descriptor)
        self.amp = amp
        self.inputs = RunningMoments(
            descriptor.state_dim + descriptor.actions + descriptor.context_dim
        )
        self.deltas = RunningMoments(descriptor.delta_dim, 1e-4)
        self.register_buffer("error_floor", self.codec.floors())
        layers = []
        width = self.inputs.mean.numel()
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.LayerNorm(size), nn.SiLU()))
            width = size
        layers.append(nn.Linear(width, descriptor.delta_dim + 4 * descriptor.contacts))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[
                descriptor.delta_dim : descriptor.delta_dim + descriptor.contacts
            ] = -3

    @torch.no_grad()
    def observe(self, x, u, context, x1):
        self.inputs.update(torch.cat((x, u, context), -1))
        self.deltas.update(self.codec.difference(x1, x))

    def forward(self, x, u, context):
        features = self.inputs(torch.cat((x, u, context), -1))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            y = self.net(features)
        y = y.float()
        delta, logits, points = y.split(
            (self.d.delta_dim, self.d.contacts, 3 * self.d.contacts), -1
        )
        x1 = self.codec.integrate(x, delta * self.deltas.std() + self.deltas.mean)
        shape = (*x.shape[:-1], self.d.hands, self.d.objects, self.d.links)
        return x1, logits.reshape(shape), points.reshape(*shape, 3)
