import Activations as act

class Fully_Connected:
    def __init__(self, size, activation):
        self.size = size
        if activation == 'ReLU':
            self.activation = act.reLu
        else if activation == 'sigmoid':
            self.activation = act.sigmoid

    def forward_pass():
        pass

    def backward_pass():
        pass

