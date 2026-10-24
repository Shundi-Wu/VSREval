"""`resample` must equal FlowNet2's `Resample2d(kernel_size=1, bilinear=True)` (a compiled CUDA operator that E*warp used originally).

The reference below is the index arithmetic of its CUDA kernel (resample2d_kernel.cu, `kernel_resample2d_update_output`): the weights come
from the fractional part of the *unclamped* sampling position, the four neighbour indices are clamped into the image.
"""
import torch

from vsreval.metrics.ewarp import resample


def kernel_reference(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    _, c, h, w = img.shape
    out = torch.zeros_like(img)
    for y in range(h):
        for x in range(w):
            xf, yf = x + float(flow[0, 0, y, x]), y + float(flow[0, 1, y, x])
            alpha, beta = xf - torch.floor(torch.tensor(xf)).item(), yf - torch.floor(torch.tensor(yf)).item()
            xl = max(min(int(torch.floor(torch.tensor(xf))), w - 1), 0)
            xr = max(min(int(torch.floor(torch.tensor(xf))) + 1, w - 1), 0)
            yt = max(min(int(torch.floor(torch.tensor(yf))), h - 1), 0)
            yb = max(min(int(torch.floor(torch.tensor(yf))) + 1, h - 1), 0)
            out[0, :, y, x] = ((1 - alpha) * (1 - beta) * img[0, :, yt, xl] + alpha * (1 - beta) * img[0, :, yt, xr]
                               + (1 - alpha) * beta * img[0, :, yb, xl] + alpha * beta * img[0, :, yb, xr])
    return out


def test_resample_matches_cuda_kernel_semantics():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 3, 17, 23, generator=g)
    # displacements up to +-30 px: most positions fall outside the image on some side
    flow = (torch.rand(1, 2, 17, 23, generator=g) - 0.5) * 60
    torch.testing.assert_close(resample(img, flow), kernel_reference(img, flow), atol=1e-5, rtol=0)


def test_zero_flow_is_identity():
    img = torch.rand(1, 3, 9, 11)
    torch.testing.assert_close(resample(img, torch.zeros(1, 2, 9, 11)), img, atol=1e-6, rtol=0)
