import torch
from typing import Optional, Union


class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]


    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((z & mask) << (2 * depth))
                | ((x & mask) << depth)
                | (y & mask)
            )
        return key


_key_lut = KeyLUT()

def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    if depth > 8:
        x_l = EX[x & mask]
        y_l = EY[y & mask]
        z_l = EZ[z & mask]
        mask = (1 << (depth - 8)) - 1
        x_h = EX[(x >> 8) & mask] << 8
        y_h = EY[(y >> 8) & mask] << 8
        z_h = EZ[(z >> 8) & mask] << 8
        x_all = x_h | x_l
        y_all = y_h | y_l
        z_all = z_h | z_l
        key = z_all << (2 * (depth - 8)) | x_all << (depth - 8) | y_all
    else:
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key
