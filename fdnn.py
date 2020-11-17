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

# CUDA_DEVICE = torch.device("cuda:0")
CUDA_DEVICE = torch.device('cpu')


def random_complex_weight_initializer(dims, device=CUDA_DEVICE):
    A = 0.01 * torch.randn(dims, device=device)
    B = 0.01 * torch.randn(dims, device=device)
    return (A, B)


def random_hadamard_filter_initializer(dims, device=CUDA_DEVICE):
    return (
        0.01 * torch.randn(dims, device=device),
        0.01 * torch.randn(dims, device=device))


def naive_bias_initializer(dims, device=CUDA_DEVICE):
    return torch.cat((torch.ones(dims), torch.ones(dims)))


class ComplexLinear(torch.nn.Module):


    def __init__(
        self, in_features, out_features,
        initializer=random_complex_weight_initializer, 
        layer_ind=0,
        bias_initializer=None, device=CUDA_DEVICE):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.Wr, self.Wi = torch.nn.Parameter(
            initializer((out_features, in_features), device))
        self.register_parameter('Wr{}'.format(layer_ind), self.Wr)
        self.register_parameter('Wi{}'.format(layer_ind), self.Wi)

        self.Br = None
        self.Bi = None

        if bias_initializer is not None:
            self.Br, self.Bi = bias_initializer()
            self.register_parameter('Br{}'.format(layer_ind), self.Br)
            self.register_parameter('Bi{}'.format(layer_ind), self.Bi)


    # @staticmethod
    def forward(self, x):
        real_ind = int(x.shape[0] / 2)
        rr = torch.nn.functional.linear(x[:real_ind], self.Wr)
        ri = torch.nn.functional.linear(x[:real_ind], self.Wi)
        ir = torch.nn.functional.linear(x[real_ind:], self.Wr)
        ii = torch.nn.functional.linear(x[real_ind:], self.Wi)

        return torch.cat((rr - ii + self.Br, ir + ri + self.Bi))


class GenericLinear(ComplexLinear):

    def __init__(
        self, num_dims, mixed_dim, in_features, out_features, layer_ind=0,
        initializer=random_complex_weight_initializer, bias_initializer=None,
        device=CUDA_DEVICE):

        super().__init__(
            in_features, out_features, layer_ind=layer_ind, 
            initializer=initializer, bias_initializer=bias_initializer,
            device=device)

        self.num_dims = num_dims
        self.mixed_dim = mixed_dim
        self.in_features = in_features
        self.out_features = out_features

        self.permutation = torch.arange(num_dims)
        self.permutation[mixed_dim] = num_dims - 1
        self.permutation[-1] = mixed_dim


    # @staticmethod
    def forward(self, x):
        """ Apply a generic linear operator on a permutation of x and return
            the un-permuted version.

        Args:
            x (torch.Tensor): Input.

        Returns:
            Tensor: Linear output with self.mixed_dim modified.
        """
        y = super().foward(x.permute(self.permutation))
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

        freal, fimag = initializer(dims, self.device)
        self.freal = torch.nn.Parameter(freal)
        self.fimag = torch.nn.Parameter(fimag)

        self.register_parameter('freal{}'.format(layer_ind), self.freal)
        self.register_parameter('fimag{}'.format(layer_ind), self.fimag)

        # self.requires_grad_(self.filter)
        self.mask = torch.ones(dims, dtype=torch.cfloat, device=device) 


    # @staticmethod
    def forward(self, x):
        r = torch.zeros_like(x, device=self.device)
        l = int(x.shape[0] / 2)
        r[:l] = self.freal * x[:l] - self.fimag * x[l: ]
        r[l:] = self.fimag * x[l:] + self.freal * x[:l]
        return r


class Absolute(torch.nn.Module):


    def __init__(self):
        super().__init__()


    # @staticmethod
    def forward(self, x):
        return x.abs()


class FrequencyFilteringBlock():


    def __init__(
        self, dims, num_layers, num_filters, non_lin=torch.nn.Tanh,
        preserved_dim=None, initializer=random_complex_weight_initializer, 
        hadamard_initializer=random_hadamard_filter_initializer,
        bias_initializer=None,
        device=CUDA_DEVICE):

        self.dims = dims
        self.num_dims = len(dims)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.initializer = initializer
        self.hadamard_initializer = hadamard_initializer
        self.bias_initializer = bias_initializer
        self.device = device

        self.preserved_dim = preserved_dim
        if preserved_dim is not None:
            self.preserved_size = dims[preserved_dim]

        self.non_linearity = non_lin
        self.layers = self.compile_layers()


    def compile_layers(self):
        layers = list()

        for l in range(self.num_layers):

            # layers.append(
            #     ComplexLinear(self.num_filters[l], self.num_filters[l + 1], 
            #     layer_ind=l, initializer=self.initializer,
            #     bias_initializer=self.bias_initializer, device=self.device))

            layers.append(
                Hadamard(self.dims[1:], initializer=self.hadamard_initializer,
                layer_ind=l, device=self.device))

            # if self.preserved_dim is not None:
            #     layers.append(
            #         GenericLinear(
            #             self.num_dims, self.preserved_dim, self.preserved_size,
            #             self.preserved_size, layer_ind=l, initializer=self.initializer,
            #             bias_initializer=self.bias_initializer,
            #             device=self.device))

            layers.append(self.non_linearity())

        return layers



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