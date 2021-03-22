from torch import nn


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=400,
        hidden_layer_num=2,
        hidden_dropout_prob=.5,
        input_dropout_prob=.2,
    ):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers
        layers = [nn.Linear(self.input_size, self.hidden_size),
                  nn.ReLU(),
                  nn.Dropout(self.input_dropout_prob)]
        for layer in range(self.hidden_layer_num):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.input_dropout_prob))
        # output
        if output_size is not None:
            layers.append(nn.Linear(self.hidden_size, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.view(len(x), -1))
