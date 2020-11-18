"""
Frequency-Domain Neural Network class for supervised classification. Each one
of the major blocks in the network will be defined in a module. The '
fundamental building blocks for this architecture are:
    - FFT layer (non-differentiable).
    - Hadamard product between complex tensors. (weight tensor to be learned)
    - Weighted sum of complex tensors. (weights to be learned)
    - The prime frequency dropin mechanism.
"""

import numpy as np
import torch
import torch.fft as fft

CUDA_DEVICE = torch.device("cuda:0")
# CUDA_DEVICE = torch.device('cpu')


def random_complex_weight_initializer(dims, alpha=0.01, device=CUDA_DEVICE):
    A = alpha * torch.randn(dims, device=device, requires_grad=True)
    B = alpha * torch.randn(dims, device=device, requires_grad=True)
    return (A, B)


def random_hadamard_filter_initializer(
    dims, alpha=1, size=None, offset=0, device=CUDA_DEVICE):

    if size is not None:
        f = torch.zeros(dims, device=device)
        f[offset : offset + size, offset : offset + size] = alpha * torch.randn(
            *dims[2:], device=device)

        y = fft.fftn(f, dim=(0, 1))
        freal = y.real
        fimag = y.imag

        freal.requires_grad_(True)
        fimag.requires_grad_(True)

        return (freal, fimag)

    return (
        alpha * torch.randn(dims, device=device, requires_grad=True),
        alpha * torch.randn(dims, device=device, requires_grad=True))


def naive_bias_initializer(dims, device=CUDA_DEVICE):
    return (torch.zeros(dims, device=device, requires_grad=True),
            torch.zeros(dims, device=device, requires_grad=True))


class ComplexBatchNorm(torch.nn.Module):


    def __init__(self, num_dims, batch_dim, batch_dim_ind, device=CUDA_DEVICE):

        super().__init__()
        self.num_dims = num_dims + 1
        permutation = list(range(self.num_dims))
        permutation[2] = batch_dim_ind + 1
        permutation[batch_dim_ind + 1] = 2
        self.permutation = tuple(permutation)

        if num_dims == 5:
            self.batchnorm = torch.nn.BatchNorm3d(batch_dim)
        else:
            self.batchnorm = torch.nn.BatchNorm2d(batch_dim)
        self.batchnorm.cuda(device)


    def forward(self, x):
        y = x.permute(self.permutation)
        return torch.stack((
            self.batchnorm(y[0]), self.batchnorm(y[1]))).permute(
                self.permutation)


class ComplexLinear(torch.nn.Module):


    def __init__(
        self, in_features, out_features,
        initializer=random_complex_weight_initializer, 
        layer_ind=0,
        bias_initializer=naive_bias_initializer, device=CUDA_DEVICE):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        Wr, Wi = initializer((out_features, in_features), device=device)
        self.Wr = torch.nn.Parameter(Wr)
        self.Wi = torch.nn.Parameter(Wi)
        self.register_parameter('Wr{}'.format(layer_ind), self.Wr)
        self.register_parameter('Wi{}'.format(layer_ind), self.Wi)

        Br, Bi = bias_initializer(out_features, device=device)
        self.Br = torch.nn.Parameter(Br)
        self.Bi = torch.nn.Parameter(Bi)
        if bias_initializer is not None:
            self.register_parameter('Br{}'.format(layer_ind), self.Br)
            self.register_parameter('Bi{}'.format(layer_ind), self.Bi)


    # @staticmethod
    def forward(self, x):
        rr = torch.nn.functional.linear(x[0], self.Wr)
        ri = torch.nn.functional.linear(x[0], self.Wi)
        ir = torch.nn.functional.linear(x[1], self.Wr)
        ii = torch.nn.functional.linear(x[1], self.Wi)

        return torch.stack((rr - ii + self.Br, ir + ri + self.Bi))


class GenericLinear(ComplexLinear):

    def __init__(
        self, num_dims, mixed_dim, in_features, out_features, layer_ind=0,
        initializer=random_complex_weight_initializer,
        bias_initializer=naive_bias_initializer,
        device=CUDA_DEVICE):

        super().__init__(
            in_features, out_features, layer_ind=layer_ind, 
            initializer=initializer, bias_initializer=bias_initializer,
            device=device)

        self.num_dims = num_dims + 1
        self.mixed_dim = mixed_dim + 1
        self.in_features = in_features
        self.out_features = out_features

        permutation = list(range(self.num_dims))
        permutation[self.mixed_dim] = self.num_dims - 1
        permutation[-1] = self.mixed_dim
        self.permutation = tuple(permutation)


    # @staticmethod
    def forward(self, x):
        """ Apply a generic linear operator on a permutation of x and return
            the un-permuted version.

        Args:
            x (torch.Tensor): Input.

        Returns:
            Tensor: Linear output with self.mixed_dim modified.
        """
        y = super().forward(x.permute(self.permutation))
        return y.permute(self.permutation)


class Hadamard(torch.nn.Module):


    def __init__(
        self, dims, layer_ind=0, initializer=random_complex_weight_initializer,
        mask_initializer=None,
        device=CUDA_DEVICE):

        super().__init__()
        self.dims = dims
        self.initializer = initializer
        self.device = device
        self.layer_ind = layer_ind

        freal, fimag = initializer(dims, device=self.device)
        self.freal = torch.nn.Parameter(freal)
        self.fimag = torch.nn.Parameter(fimag)
        self.register_parameter('freal{}'.format(layer_ind), self.freal)
        self.register_parameter('fimag{}'.format(layer_ind), self.fimag)

        # apply mask only if necessary
        if mask_initializer is not None:
            self.mask = mask_initializer(dims, device=device)
            self.ff = self.masked_picewise_prod
        else:
            self.ff = self.picewise_prod


    # @staticmethod
    def forward(self, x):
        return self.ff(x)


    def picewise_prod(self, x):
        return torch.stack((
            self.freal * x[0] - self.fimag * x[1],
            self.fimag * x[1] + self.freal * x[0]))


    def masked_picewise_prod(self, x):
        return self.mask * self.picewise_prod(x)


class Absolute(torch.nn.Module):


    def __init__(self):
        super().__init__()


    # @staticmethod
    def forward(self, x):
        return x.square().sum(0).sqrt()



class FrequencyFilteringBlock(torch.nn.Module):


    def __init__(
        self, dims, num_filters, non_lin=torch.nn.Hardtanh,
        preserved_dim=None, initializer=random_complex_weight_initializer, 
        hadamard_initializer=random_hadamard_filter_initializer,
        bias_initializer=naive_bias_initializer,
        device=CUDA_DEVICE,
        dropout=None):

        super().__init__()
        self.dims = dims
        self.num_dims = len(dims)
        self.num_layers = int(len(num_filters) - 1)
        self.num_filters = num_filters
        self.initializer = initializer
        self.hadamard_initializer = hadamard_initializer
        self.bias_initializer = bias_initializer
        self.device = device

        self.preserved_dim = preserved_dim
        self.is_preserving = preserved_dim is not None
        if self.is_preserving:
            self.preserved_size = dims[preserved_dim]
        self.batch_dim_index = 3 # TODO: clarify why

        self.dropout = dropout
        self.non_linearity = non_lin
        self.layers = self.compile_layers()
        self.sequential = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        return self.sequential(x)


    def compile_layers(self):
        layers = list()

        for l in range(self.num_layers):

            layers.append(
                ComplexLinear(self.num_filters[l], self.num_filters[l + 1], 
                layer_ind=l, initializer=self.initializer,
                bias_initializer=self.bias_initializer, device=self.device))

            layers.append(
                Hadamard(
                    self.hadamar_dimension(self.num_filters[l + 1]),
                    initializer=self.hadamard_initializer,
                    layer_ind=l, device=self.device))

            if self.is_preserving:
                layers.append(
                    GenericLinear(
                        self.num_dims, self.preserved_dim, self.preserved_size,
                        self.preserved_size, layer_ind=l,
                        initializer=self.initializer,
                        bias_initializer=self.bias_initializer,
                        device=self.device))

            layers.append(self.non_linearity())
            layers.append(ComplexBatchNorm(
                self.num_dims, self.batch_norm_dims(l + 1),
                self.batch_dim_index))

            if self.dropout is not None:
                layers.append(torch.nn.Dropout(self.dropout))

        return layers


    def hadamar_dimension(self, num_filters, preserving=False):
        if self.is_preserving:
            return (*self.dims[1: -1], num_filters)
        return (self.dims[1], self.dims[2], num_filters)


    def batch_norm_dims(self, l):
        if self.is_preserving:
            return 3
        return self.num_filters[l]


    def num_output_features(self):
        return np.prod(self.dims[1 :-1]) * self.num_filters[-1]
    

    def output_shape(self):
        return tuple(*self.dims[1:-1], self.num_filters[-1])


class ComplexClassificationHead(torch.nn.Module):

    def __init__(self, in_features, num_classes, device=CUDA_DEVICE):
        
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.device = device

        self.layers = [
            torch.nn.Flatten(start_dim=2),
            ComplexLinear(in_features, num_classes, device=device), # standard linear func
            Absolute()]
        
        self.sequential = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        
        return self.sequential(x)
    

class FourierPreprocess(torch.nn.Module):


    def __init__(self, perm=(0, 2, 3, 1), fourier_dim=(1, 2), append_dim=1):

        super().__init__()
        self.p = perm
        self.fdim = fourier_dim
        self.append = append_dim


    def forward(self, x):
        with torch.no_grad():
            x = x.permute(self.p)
            x = fft.fftn(x, dim=self.fdim)
            x = x.view((*x.shape, self.append))
            x = torch.stack((x.real, x.imag))
        return x



class ModReLU(torch.nn.Module):

    def __init__(
        self, dims, bias_initializer=naive_bias_initializer, layer_ind=0,
        device=CUDA_DEVICE):

        super().__init__()
        self.dims = dims
        self.abs = Absolute()

        bias = bias_initializer(dims, device=device)
        self.bias = torch.nn.Parameter(bias)
        self.register_parameter('rb{}'.format(layer_ind), self.bias)


    def forward(self, x):
        mod = 1 / self.abs(x)
        mask = (1 + (self.bias * mod)) > 0
        return mask.view((1, 1, *x.size())) * x


class FrequencyDomainNeuralNet(torch.nn.Module):

    def __init__(self, pdims, p_num_filters, m_num_filters, num_classes,
        p_non_lin=torch.nn.Hardtanh, m_non_lin=torch.nn.Hardtanh,
        preserved_dim=3, p_initializer=random_complex_weight_initializer,
        m_initializer=random_complex_weight_initializer, 
        p_hadamard_initializer=random_hadamard_filter_initializer,
        m_hadamard_initializer=random_hadamard_filter_initializer,
        p_bias_initializer=naive_bias_initializer,
        m_bias_initializer=naive_bias_initializer,
        collapse_initializer=random_complex_weight_initializer,
        dropout=None, 
        device=CUDA_DEVICE):

        super().__init__()
        self.num_dims = len(pdims)

        self.preserving_block = FrequencyFilteringBlock(
                    pdims, p_num_filters, non_lin=p_non_lin,
                    preserved_dim=preserved_dim, 
                    initializer=p_initializer, 
                    hadamard_initializer=p_hadamard_initializer,
                    bias_initializer=p_bias_initializer,
                    device=device, dropout=dropout)

        self.mdims = (*pdims[:-2], p_num_filters[-1])
        self.device = device

        preserved_size = pdims[preserved_dim]

        self.collapse = GenericLinear(
                        len(pdims), preserved_dim, preserved_size,
                        1, layer_ind=-1,
                        initializer=collapse_initializer,
                        bias_initializer=p_bias_initializer,
                        device=device)

        # self.collapse_view_dims = self.mdims[1:]

        self.mixing_block = FrequencyFilteringBlock(
                    self.mdims, 
                    (p_num_filters[-1],  *m_num_filters),
                    non_lin=m_non_lin,
                    preserved_dim=None, 
                    initializer=m_initializer, 
                    hadamard_initializer=m_hadamard_initializer,
                    bias_initializer=m_bias_initializer,
                    device=device, dropout=dropout)

        self.head = ComplexClassificationHead(
            self.mixing_block.num_output_features(), num_classes, device=device)


    def forward(self, x):
        y0 = self.preserving_block(x)
        y1 = self.collapse(y0).view((x.shape[0], x.shape[1], *self.mdims[1:]))
        return self.head(self.mixing_block(y1))
