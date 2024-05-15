import torch
from torch import nn

class MorphologicalMax(nn.Module):
    def __init__(self, in_units, dil_units, ero_units, lin_units):
        super().__init__()

        self.dilation_weights = nn.Parameter(torch.randn(dil_units, in_units + 1))
        self.erosion_weights = nn.Parameter(torch.randn(ero_units, in_units + 1))
        self.linear_weights = nn.Parameter(torch.randn(lin_units, dil_units + ero_units))

    def forward(self, input_batch):

        # Input: Fix dimensions
        # Otherwise, the addition & subtraction
        # do not work as expected
        batch_size, image_size = input_batch.shape
        X = input_batch.reshape((batch_size, 1, image_size))

        # Required for the delimiters: Add a zero to every
        # element of the input batch
        X_bar = torch.cat(
            [X, torch.zeros((batch_size, 1, 1), dtype=X.dtype)],
            dim=-1
        )

        dilation_result = torch.max(
            # X_bar + dilation_weights
            torch.add(X_bar, self.dilation_weights),
            # Compute the maximum in dimension 2
            # Note: The batch dimension is dimension 0!
            2
        )
        erosion_result = torch.min(
            # X_bar - erosion_weights
            torch.sub(X_bar, self.erosion_weights),
            # Compute the minimum in the correct dimension
            2
        )

        return torch.matmul(
            # Multiply the linear weight matrix...
            self.linear_weights,
            # ... with a concatenation of the previous results.
            torch.cat(
                [dilation_result.values, erosion_result.values],
                # Do the concatenation in the correct dimension
                1
                # ... and transpose the thing.
            ).t()
        # Transpose the result again, so that in the end, I
        # receive the correct output
        ).t()
