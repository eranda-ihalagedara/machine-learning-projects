class Model:
    # Set layers
    def __init__(self, layers):
        self.layers = layers

    # Build each layer
    def build(self):
        size_l = layers[0].size_out
        layers[0].build(size_l)
        for i in range(1, len(self.layers)):
            layers[i].build(size_l)
            size_l = layers[i].size_out

    # Train model
    def train(self):
        pass

    # Predict - forward pass through each layer
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
        
        