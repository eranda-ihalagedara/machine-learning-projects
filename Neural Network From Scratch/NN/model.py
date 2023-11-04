class Model:
    # Set layers
    def __init__(self, layers, learning_rate=0.001, opt = 'gradient_desccent'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.opt = opt
        self.build()

    # Build each layer
    def build(self):
        size_l = self.layers[0].size_out
        self.layers[0].build(size_l)
        for i in range(1, len(self.layers)):
            self.layers[i].build(size_l)
            size_l = self.layers[i].size_out

    # Train model
    def train(self, x_train, y_train, learning_rate=0.001):
        pass

    # Predict - forward pass through each layer
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
        
        