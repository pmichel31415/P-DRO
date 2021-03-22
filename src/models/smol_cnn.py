from torch import nn


class CNN(nn.Module):
    """A smol convolutional net"""

    def __init__(
        self, input_shape,
        output_size,
        kernels=None,
        hidden_dropout_prob=.5,
        input_dropout_prob=.2,
        pool_every=2,
    ):
        # Configurations.
        super().__init__()
        default_kernels = [(3, 32), (3, 32), (3, 64), (3, 64), (3, 512)]
        self.kernels = kernels or default_kernels
        self.input_shape = input_shape
        self.input_dropout_prob = input_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        shape = list(input_shape)
        # Layers
        layers = [nn.Dropout(self.input_dropout_prob)]
        for layer in range(len(self.kernels)):
            kernel_size, out_channels = self.kernels[layer]
            layers.append(
                nn.Conv2d(
                    shape[0],
                    out_channels,
                    kernel_size,
                    padding=1
                )
            )
            shape[0] = out_channels
            layers.append(nn.ReLU())
            if (layer+1) % pool_every == 0:
                layers.append(nn.MaxPool2d(2, 2))
                shape[1:] = [shape[1] // 2, shape[2] // 2]
            layers.append(nn.Dropout(self.hidden_dropout_prob))
        self.layers = nn.Sequential(*layers)
        self.hidden_size = shape[0] * shape[1] * shape[2]
        # Output
        self.output_layer = None
        if output_size is not None:
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        bsz = len(x)
        h = self.layers(x).view(bsz, -1)
        if self.output_layer is not None:
            h = self.output_layer(h)
        return h
