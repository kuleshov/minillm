import numpy as np
import torch
import torch.nn as nn
from minillm.engine.gptq import quantize

try:
    import quant_cuda
except:
    print('CUDA extension not installed. Inference will not work.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(self, bits, infeatures, outfeatures):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.bits = bits
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int)
        )

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()  

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 256 * (self.bits * 8), intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        outshape = list(x.shape)
        x = x.reshape(-1, x.shape[-1])
        y = self.bias.clone().repeat(x.shape[0],1)
        outshape[-1] = self.bias.numel()
        dtype = x.dtype
        x = x.float()
        if self.bits == 2:
            quant_cuda.vecquant2matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 3:
            quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 4:
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)
        elif self.bits == 8:
            quant_cuda.vecquant8matmul(x, self.qweight, y, self.scales, self.zeros)
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(dtype)
        return y.reshape(outshape)
