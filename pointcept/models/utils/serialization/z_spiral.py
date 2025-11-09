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
        xy_key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            xy_key = (
                    xy_key
                    | ((x & mask) << depth)
                    | (y & mask)
            )
        for i in range(depth):
            mask = 1 << i
            mask_dual = 1 << (i + depth)
            key = (
                    key
                    | ((z & mask) << (2 * i + 2))
                    | ((xy_key & mask_dual) << i)
                    | ((xy_key & mask) << i)
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
    if depth > 8:
        key = _key_lut.xyz2key(x, y, z, depth)
    else:
        EX, EY, EZ = _key_lut.encode_lut(x.device)
        x, y, z = x.long(), y.long(), z.long()

        mask = 255 if depth > 8 else (1 << depth) - 1
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key


