import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_
from .a_spiral import xyz2key as a_spiral_encode_
from .z_spiral import xyz2key as z_spiral_encode_
from .h_spiral import encode_3d_spiral_hilbert as h_spiral_encode_

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "a_s", "z_s", "h_s", "h_s_t", "z_s_t",
                     "hilbert-q90", "hilbert-q270",
                     "z-trans-q90", "z-trans-q180", "z-trans-q270",
                     "z-q90", "z-q180", "z-q270",
                     "hilbert-trans-q90", "hilbert-trans-q270"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "z-q90":
        code = z_order_encode_q90(grid_coord, depth=depth)
    elif order == "z-q180":
        code = z_order_encode_q180(grid_coord, depth=depth)
    elif order == "z-q270":
        code = z_order_encode_q270(grid_coord, depth=depth)
    elif order == "z-trans-q90":
        code = z_order_encode_q90(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "z-trans-q180":
        code = z_order_encode_q180(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "z-trans-q270":
        code = z_order_encode_q270(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-q90":
        code = hilbert_encode_q90(grid_coord, depth=depth)
    elif order == "hilbert-q180":
        code = hilbert_encode_q180(grid_coord, depth=depth)
    elif order == "hilbert-q270":
        code = hilbert_encode_q270(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert-trans-q90":
        code = hilbert_encode_trans_q90(grid_coord, depth=depth)
    elif order == "hilbert-trans-q270":
        code = hilbert_encode_trans_q270(grid_coord, depth=depth)
    elif order == "a_s":
        code = a_spiral_encode(grid_coord, depth=depth)
    elif order == "z_s":
        code = z_spiral_encode(grid_coord, depth=depth)
    elif order == "z_s_t":
        code = z_spiral_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "h_s":
        code = h_spiral_encode(grid_coord[:, [2, 0, 1]], depth=depth)
    elif order == "h_s_t":
        code = h_spiral_encode(grid_coord[:, [2, 1, 0]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_encode_q90(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    y_vals = y.clone()

    low = torch.quantile(y_vals.float(), 0.25).round().long()  # 1/4 位置
    y_max = y_vals.max()

    y_vals += low
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y_vals, z, b=None, depth=depth)
    return code

def z_order_encode_q180(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    y_vals = y.clone()

    low = y_vals.median()
    y_max = y_vals.max()

    y_vals += low
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y_vals, z, b=None, depth=depth)
    return code

def z_order_encode_q270(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    y_vals = y.clone()

    low = torch.quantile(y_vals.float(), 0.75).round().long()  # 1/4 位置
    y_max = y_vals.max()

    y_vals += low
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y_vals, z, b=None, depth=depth)
    return code
def z_order_decode(code: torch.Tensor, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)

def hilbert_encode_q90(grid_coord: torch.Tensor, depth: int = 16):
    coord = grid_coord.clone()
    y_vals = coord[:, 1]

    low = torch.quantile(y_vals.float(), 0.25).round().long()  # 1/4 位置
    y_max = y_vals.max()

    y_vals += low
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    coord[:, 1] = y_vals
    return hilbert_encode_(coord, num_dims=3, num_bits=depth)

def hilbert_encode_q180(grid_coord: torch.Tensor, depth: int = 16):
    y_vals = grid_coord[:, 1].clone()

    median = y_vals.median()
    y_max = y_vals.max()

    y_vals += median
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    coord = torch.empty_like(grid_coord)
    coord[:, [0, 2]] = grid_coord[:, [0, 2]]
    coord[:, 1] = y_vals
    return hilbert_encode_(coord, num_dims=3, num_bits=depth)

def hilbert_encode_trans_q270(grid_coord: torch.Tensor, depth: int = 16):
    y_vals = grid_coord[:, 0].clone()

    high = torch.quantile(y_vals.float(), 0.75).round().long()  # 1/4 位置
    y_max = y_vals.max()

    y_vals += high
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    coord = torch.empty_like(grid_coord)
    coord[:, [1, 2]] = grid_coord[:, [1, 2]]
    coord[:, 0] = y_vals
    return hilbert_encode_(coord, num_dims=3, num_bits=depth)

def hilbert_encode_trans_q90(grid_coord: torch.Tensor, depth: int = 16):
    y_vals = grid_coord[:, 0].clone()

    low = torch.quantile(y_vals.float(), 0.25).round().long()  # 1/4 位置
    y_max = y_vals.max()

    y_vals += low
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    coord = torch.empty_like(grid_coord)
    coord[:, [1, 2]] = grid_coord[:, [1, 2]]
    coord[:, 0] = y_vals
    return hilbert_encode_(coord, num_dims=3, num_bits=depth)

def hilbert_encode_q270(grid_coord: torch.Tensor, depth: int = 16):
    coord = grid_coord.clone()
    y_vals = coord[:, 1]

    high = torch.quantile(y_vals.float(), 0.75).round().long()  # 3/4 位置
    y_max = y_vals.max()

    y_vals += high
    y_vals = torch.where(y_vals > y_max, y_vals - y_max, y_vals)
    coord[:, 1] = y_vals
    return hilbert_encode_(coord, num_dims=3, num_bits=depth)

def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)

def a_spiral_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = a_spiral_encode_(x, y, z, b=None, depth=depth)
    return code

def z_spiral_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_spiral_encode_(x, y, z, b=None, depth=depth)
    return code

def h_spiral_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    grid_coord_xyz = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return h_spiral_encode_(grid_coord_xyz, m_xy=depth * 2, m_z=depth)