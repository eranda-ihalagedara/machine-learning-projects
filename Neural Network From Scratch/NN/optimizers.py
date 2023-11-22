import numpy as np

class rmsprop:
    def __init__(self, beta, w_shape, b_shape) -> None:
        self.beta = beta
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

class adam:
    def __init__(self) -> None:
        pass