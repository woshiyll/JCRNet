import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.ops import functional as F

class CharbonnierLoss(nn.Cell):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def construct(self, x, y):
        diff = x - y
        loss = ops.mean(ops.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Cell):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = ops.tile(ops.matmul(k.t(),k).unsqueeze(0),(3,1,1,1))
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = ops.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def construct(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
