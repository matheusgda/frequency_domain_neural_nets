"""
Frequency-Domain Neural Network class for supervised classification. Each one
of the major blocks in the network will be defined in a module. The '
fundamental building blocks for this architecture are:
    - FFT layer (non-differentiable).
    - Hadamard product between complex tensors. (weight tensor to be learned)
    - Weighted sum of complex tensors. (weights to be learned)
    - The prime frequency dropin mechanism.
"""

import torch
import numpy as np

CUDA_DEVICE = torch.device("cuda:0")
# CUDA_DEVICE = torch.device('cpu')


def random_complex_weight_initializer(dims, device=CUDA_DEVICE):
    A = 0.01 * torch.randn(dims, device=device, requires_grad=True)
    B = 0.01 * torch.randn(dims, device=device, requires_grad=True)
    return (A, B)


def random_hadamard_filter_initializer(dims, device=CUDA_DEVICE):
    return (
        1 * torch.randn(dims, device=device, requires_grad=True),
        1 * torch.randn(dims, device=device, requires_grad=True))


def naive_bias_initializer(dims, device=CUDA_DEVICE):
    return (torch.zeros(dims, device=device, requires_grad=True),
            torch.zeros(dims, device=device, requires_grad=True))


class ComplexBatchNorm(torch.nn.Module):


    def __init__(self, num_dims, batch_dim, batch_dim_ind, device=CUDA_DEVICE):
        super().__init__()
        permutation = list(range(num_dims))
        permutation[1] = batch_dim_ind
        permutation[batch_dim_ind] = 1
        self.permutation = tuple(permutation)

        if num_dims == 5:
            self.batchnorm = torch.nn.BatchNorm3d(batch_dim)
        else:
            self.batchnorm = torch.nn.BatchNorm2d(batch_dim)
        self.batchnorm.cuda(device)
        print(self.permutation)


    def forward(self, x):
        rind = int(x.shape[0] / 2)
        y = x.permute(self.permutation)
        return torch.cat((
            self.batchnorm(y[:rind]), self.batchnorm(y[rind:]))).permute(
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

        Wr, Wi = initializer((out_features, in_features), device)
        self.Wr = torch.nn.Parameter(Wr)
        self.Wi = torch.nn.Parameter(Wi)
        self.register_parameter('Wr{}'.format(layer_ind), self.Wr)
        self.register_parameter('Wi{}'.format(layer_ind), self.Wi)

        Br, Bi = bias_initializer(out_features, device)
        self.Br = torch.nn.Parameter(Br)
        self.Bi = torch.nn.Parameter(Bi)
        if bias_initializer is not None:
            self.register_parameter('Br{}'.format(layer_ind), self.Br)
            self.register_parameter('Bi{}'.format(layer_ind), self.Bi)


    # @staticmethod
    def forward(self, x):
        rind = int(x.shape[0] / 2)
        rr = torch.nn.functional.linear(x[:rind], self.Wr)
        ri = torch.nn.functional.linear(x[:rind], self.Wi)
        ir = torch.nn.functional.linear(x[rind:], self.Wr)
        ii = torch.nn.functional.linear(x[rind:], self.Wi)

        return torch.cat((rr - ii + self.Br, ir + ri + self.Bi))


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

        self.num_dims = num_dims
        self.mixed_dim = mixed_dim
        self.in_features = in_features
        self.out_features = out_features

        permutation = list(range(num_dims))
        permutation[mixed_dim] = num_dims - 1
        permutation[-1] = mixed_dim
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
        mask_initializer=torch.ones,
        device=CUDA_DEVICE):

        super().__init__()
        self.dims = dims
        self.initializer = initializer
        self.device = device
        self.layer_ind = layer_ind
        print(self.dims)

        freal, fimag = initializer(dims, self.device)
        self.freal = torch.nn.Parameter(freal)
        self.fimag = torch.nn.Parameter(fimag)
        self.register_parameter('freal{}'.format(layer_ind), self.freal)
        self.register_parameter('fimag{}'.format(layer_ind), self.fimag)

        self.mask = torch.ones(dims, dtype=torch.cfloat, device=device) 


    # @staticmethod
    def forward(self, x):
        rind = int(x.shape[0] / 2)
        return torch.cat((
            self.freal * x[:rind] - self.fimag * x[rind:],
            self.fimag * x[rind:] + self.freal * x[:rind]))


class Absolute(torch.nn.Module):


    def __init__(self):
        super().__init__()


    # @staticmethod
    def forward(self, x):
        rind = int(x.shape[0] / 2)
        return (x[:rind] * x[:rind] + x[rind:] * x[rind:]).sqrt()


class FrequencyFilteringBlock(torch.nn.Module):


    def __init__(
        self, dims, num_layers, num_filters, non_lin=torch.nn.Hardtanh,
        preserved_dim=None, initializer=random_complex_weight_initializer, 
        hadamard_initializer=random_hadamard_filter_initializer,
        bias_initializer=naive_bias_initializer,
        device=CUDA_DEVICE):

        super().__init__()
        self.dims = dims
        self.num_dims = len(dims)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.initializer = initializer
        self.hadamard_initializer = hadamard_initializer
        self.bias_initializer = bias_initializer
        self.device = device

        self.preserved_dim = preserved_dim
        self.is_preserving = preserved_dim is not None
        if self.is_preserving:
            self.preserved_size = dims[preserved_dim]
        self.batch_dim_index = int (4 - self.is_preserving)

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


class ComplexClassificationHead(torch.nn.Module):

    def __init__(self, in_features, num_classes, device=CUDA_DEVICE):
        
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.device = device

        self.layers = [
            torch.nn.Flatten(),
            ComplexLinear(in_features, num_classes, device=device), # standard linear func
            Absolute()]
        
        self.sequential = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)


# class ConvFilteringBlock(torch.nn.Module):

    
#     def __init__(self, num_layers, num_filters, dims):

#         super(ConvFilteringBlock, self).__init__()
#         self.num_layers = num_layers
#         self.num_filters = num_filters
#         self.dims = dims
#         self.size = []

#         self.filters = [
#             torch.nn.Parameter(torch.rand(
#                 (num_filters, num_filters, *dims),
#                 device=device,
#                 dtype=torch.cfloat)) for _ in range(num_layers)]
#         for f in range(len(self.filters)):
#             self.register_parameter("param_{}".format(f), self.filters[f])


#     def forward(self, x):
#         x_size = x.size()
#         size = list()
#         size.append(x_size[0])
#         size.append(1)
#         for i in range (1, len(x_size)):
#             size.append(x_size[i])

#         for i in range(self.num_layers):
#             # print(x.view(tuple(size)).size(), self.filters[i].size())
#             had = x.view(tuple(size)) * self.filters[i] # (N, C, H, W) * (F, C, H, W)
#             conv = torch.sum(had, axis=1)
#         return conv


# # This class represents an NN module with all 3 major blocs from an 
# #  FDNN.
# #
# class FDNN(torch.nn.Module):

#     def __init__(
#         self, input_dims, num_chp_layers, num_clayers, num_filters, num_classes):

#         super(FDNN, self).__init__()
#         self.input_dims = input_dims
#         self.num_chp_layers = num_chp_layers
#         self.num_clayers = num_clayers
#         self.num_filters = num_filters
#         self.num_classes = num_classes
#         self.linearized_size = np.prod(input_dims[1:]) * num_filters

#         self.conv_block = ConvFilteringBlock(
#             self.num_clayers, num_filters, self.input_dims[1:])
        
#         # self.linear = torch.nn.Linear(
#         #     self.linearized_size, self.num_classes, dt)
#         self.w = torch.nn.Parameter(torch.rand(
#             self.linearized_size,
#             self.num_classes,
#             dtype=torch.float,
#             device=device))
#         self.register_parameter('w', self.w)

#     def forward(self, x):
#         conv = self.conv_block.forward(x).abs()
#         lin = conv.view((x.size()[0], self.linearized_size)) @ self.w 
#         # return torch.abs(self.linear(conv.view(x.size()[0], self.linearized_size)))
#         return lin


# class MyReLU(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input