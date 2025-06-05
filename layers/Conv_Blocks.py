import torch
import torch.nn as nn
import math

## this class models the time-aware convolutional layer
## the following is the parallel version of the previous one (which is still present in the code for reference, but commented out)
class TimeAwareConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_sequences=4, padding=0):
        super(TimeAwareConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_sequences = num_sequences
        self.padding = padding

        # Learnable kernel weights shaped for input embeddings, sequences, and kernel size
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, num_sequences, kernel_size)
        )

    def forward(self, x, time_tensor):
        """
        x: Input tensor of shape (batch_size, in_channels, num_sequences, sequence_length).
        time_tensor: Time information tensor of shape (batch_size, num_sequences, sequence_length).
        """
        batch_size, in_channels, num_sequences, sequence_length = x.shape

        # Ensure input matches the expected dimensions
        assert in_channels == self.in_channels, "Input in_channels does not match the initialized value."
        assert num_sequences == self.num_sequences, "Input num_sequences does not match the initialized value."

        # Apply padding to maintain sequence length
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x, 
                (self.padding, self.padding),  # Apply padding along the sequence dimension
                mode='constant', 
                value=0
            )
            time_tensor = torch.nn.functional.pad(
                time_tensor, 
                (self.padding, self.padding), 
                mode='constant', 
                value=0
            )
            sequence_length += 2 * self.padding

        # Unfold the input tensor to get sliding windows
        x_unfolded = x.unfold(-1, self.kernel_size, 1)  # [batch_size, in_channels, num_sequences, sequence_length - kernel_size + 1, kernel_size]
        x_unfolded = x_unfolded.permute(0, 2, 1, 4, 3)  # [batch_size, num_sequences, in_channels, kernel_size, sequence_length - kernel_size + 1]

        ### we also have to apply padding on x_unfolded
        ### because otherwise the convolution will not be able to slide over the entire sequence
        ### and the dimensions do not match with dynamic_weights
        if self.padding > 0:
            x_unfolded = torch.nn.functional.pad(
                x_unfolded, 
                (self.padding, self.padding),  # Apply padding along the sequence dimension
                mode='constant', 
                value=0
            )

        # Calculate dynamic weights for the entire sequence
        dynamic_weights = self.weights.unsqueeze(0).unsqueeze(-1) * time_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        # dynamic_weights: [batch_size, out_channels, in_channels, num_sequences, kernel_size, sequence_length]

        # Perform convolution using einsum
        # for the future: one typical error is to forget to mention each dimension of the tensors
        # e.g. if "boick,bcikl->bocl" is used, it will throw an eerror because dynamic_weights has 6 dimensions
        conv_result = torch.einsum("boickl,bcikl->bocl", dynamic_weights, x_unfolded)
        # output dimensions: [batch_size, out_channels, num_sequences, sequence_length]
        
        # Remove padding from the result if necessary
        if self.padding > 0:
            conv_result = conv_result[..., self.padding:-self.padding]
        
        return conv_result
   
# ### this is the sequential (for loop) version of the parallel TimeAwareConv1d class (keep for reference)
# class TimeAwareConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_sequences=4, padding=0):
#         """
#         in_channels: Number of input embeddings (input channels).
#         out_channels: Number of convolutional kernels (output channels).
#         kernel_size: Size of the kernel along the time dimension.
#         num_sequences: Number of sequences (spanning dimension for kernels).
#         """
#         super(TimeAwareConv1d, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.num_sequences = num_sequences
#         self.padding = padding

#         # Learnable kernel weights shaped for input embeddings, sequences, and kernel size
#         self.weights = nn.Parameter(
#             torch.randn(out_channels, in_channels, num_sequences, kernel_size)
#         )

#     def forward(self, x, time_tensor):
#         """
#         x: Input tensor of shape (batch_size, in_channels, num_sequences, sequence_length).
#         time_tensor: Time information tensor of shape (batch_size, num_sequences, sequence_length).
#         """
#         batch_size, in_channels, num_sequences, sequence_length = x.shape

#         # Ensure input matches the expected dimensions
#         assert in_channels == self.in_channels, "Input in_channels does not match the initialized value."
#         assert num_sequences == self.num_sequences, "Input num_sequences does not match the initialized value."

#         # Apply padding to maintain sequence length
#         if self.padding > 0:
#             x = torch.nn.functional.pad(
#                 x, 
#                 (self.padding, self.padding),  # Apply padding along the sequence dimension
#                 mode='constant', 
#                 value=0
#             )
#             time_tensor = torch.nn.functional.pad(
#                 time_tensor, 
#                 (self.padding, self.padding), 
#                 mode='constant', 
#                 value=0
#             )

#         # Initialize output tensor
#         output = torch.zeros(
#             batch_size, 
#             self.out_channels, 
#             num_sequences, # needs to be dropped in case of einsum "->bo" (see below)
#             sequence_length  # sequence length remains the same due to padding
#         ).to(x.device)

#         # Iterate over the time dimension for sliding kernel application
#         for t in range(sequence_length):

#             # Extract time slice (batch_size, num_sequences, kernel_size)
#             time_slice = time_tensor[:, :, t:t + self.kernel_size]  # Shape: [batch_size, num_sequences, kernel_size]

#             # Expand weights to align with batch dimensions
#             # self.weights: [out_channels, in_channels, num_sequences, kernel_size]
#             # Expanded: [batch_size, out_channels, in_channels, num_sequences, kernel_size]
#             expanded_weights = self.weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

#             # Expand time_slice to align with expanded_weights
#             # time_slice: [batch_size, num_sequences, kernel_size]
#             # Expanded: [batch_size, 1, 1, num_sequences, kernel_size]
#             expanded_time_slice = time_slice.unsqueeze(1).unsqueeze(1)  # Adds two singleton dimensions

#             # Compute dynamic weights by modulating expanded_weights with expanded_time_slice
#             # Resulting shape: [batch_size, out_channels, in_channels, num_sequences, kernel_size]
#             dynamic_weights = expanded_weights * expanded_time_slice

#             # Extract input slice corresponding to the current time step
#             x_slice = x[:, :, :, t:t + self.kernel_size]  # Shape: [batch_size, in_channels, num_sequences, kernel_size]

#             # Perform convolution across in_channels, num_sequences, and kernel_size
#             # dynamic_weights: [batch_size, out_channels, in_channels, num_sequences, kernel_size]
#             # x_slice: [batch_size, in_channels, num_sequences, kernel_size]
#             ### einsum is mainly needed because we have dynamic kernels parameterized by time
#             ### and to handle multi-dimensional interactions efficiently
#             # conv_result = torch.einsum("boick,bick->bo", dynamic_weights, x_slice)  # Shape: [batch_size, out_channels]
#             ### this is good for tasks where relationships across sequences (global relationships) are more important 
#             ### than individual sequence dynamics.
#             ### in case we want one value (activation) for each sequence in num_sequences, 
#             ### we need the following code: 
#             # conv_result = torch.einsum("boick,bick->bock", dynamic_weights, x_slice)  # Shape: [batch_size, out_channels (=num_kernels), num_sequences, kernel_size]
#             # # Reshape or squeeze kernel size dimension (if kernel_size == 1, q=1)
#             # conv_result = conv_result.squeeze(-1)  # Shape: [batch_size, out_channels, num_sequences, sequence_length]
#             ### in this case we get 4 feature maps per kernel, one for each sequence in num_sequences
#             ### suitable when each sequence's features are independently important or when sequence-specific patterns need to be preserved
#             ### intuitive analogy: Think of the sequences (c) as different data streams (e.g., multiple sensors) and kernels (o) as feature extractors
#             ### "boick,bick->bock" [batch_size, out_channels, num_sequences] gives a separate feature map for each sensor for each kernel.
#             ### "boick,bick->bo" [batch_size, out_channels] combines all sensors into a single feature map per kernel, losing sequence-specific detail.
#             ### "boick,bick->boc" [batch_size, out_channels, num_sequences] provides a sequence-aware intermediate representation for each kernel.
#             ### for the time we stick to "boc", because the consecutive Inception_Block layer 
#             ### needs to have the same number of sequences as the input
#             ### this option ("->boc" and not "->bock") is the closest match to the original inception block using nn.Conv2d
#             conv_result = torch.einsum("boick,bick->boc", dynamic_weights, x_slice)  # Shape: [batch_size, out_channels, num_sequences]
            
#             # Save result for the current time step
#             output[:, :, :, t] = conv_result

#         return output


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            # kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            kernels.append(TimeAwareConv1d(in_channels, out_channels, kernel_size=2 * i + 1, num_sequences=4, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ### attribute is called "weight" not "weights"
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, TimeAwareConv1d):
                ### atribute is called "weights" not "weight"
                nn.init.kaiming_normal_(m.weights, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, doy):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x, doy))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# class Inception_Block_V2(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super(Inception_Block_V2, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
#         kernels = []
#         #### if we want larger kernels, remove the // 2
#         for i in range(self.num_kernels // 2):
#             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
#             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
#         kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
#         self.kernels = nn.ModuleList(kernels)
#         if init_weight:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         res_list = []
#         for i in range(self.num_kernels + 1):
#             res_list.append(self.kernels[i](x))
#         res = torch.stack(res_list, dim=-1).mean(-1)
#         return res
