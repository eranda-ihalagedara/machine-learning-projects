import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import NN.losses as losses
from .softmax import Softmax
import logging

class Model:
    # Set layers
    def __init__(self, layers, learning_rate=0.0001, opt = 'SGD', loss='mean_squared_error', lr_decay=1,):
        self.layers = layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.opt = opt
        self.loss_fn = self.get_loss_fn(loss)
        self.build()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    # Build each layer
    def build(self):
        size_l = self.layers[0].size_out if self.layers[0].size_in == None else self.layers[0].size_in
        self.layers[0].build(size_l,0)
        size_l = self.layers[0].size_out
        
        for i in range(1, len(self.layers)):
            self.layers[i].build(size_l, i)
            size_l = self.layers[i].size_out

        # Default to softmax loss of last layer is softmax
        if isinstance(self.layers[-1], Softmax):
            self.loss_fn = self.get_loss_fn('categorical_cross_entropy')
            self.logger.info('Defaulting to categorical_cross_entropy')

    # Train model
    def train(self, x_train, y_train, batch_size = 32, epochs = 1, cv = None):
        m = x_train.shape[1]
        steps = np.ceil(m/batch_size)
        self.metrics_list = {}
        
        for epoch in range(epochs):
            
            for i in range(0, m, batch_size):
                x = x_train[:, i:min(i+batch_size,m)]
                a_l = self.predict(x)

                _, da_l = self.loss_fn(a_l,y_train[:, i:min(i+batch_size,m)])
                
                # Clip gradient values
                da_l = np.maximum(-1e6,np.minimum(1e6, da_l))

                for layer in list(reversed(self.layers)):
                    da_l = layer.backward_pass(da_l)
                    layer.update_weights(self.learning_rate)

                # Progress bar
                percent = np.round(50*((i//batch_size + 1)/steps),2)
                
                if percent*2 < 100 :
                    end_char = '\r'
                else:
                    end_char = ' '
                    
                print('epoch:', epoch+1,'='*percent.astype(int) + ' '*(50-percent.astype(int)),
                      '{:.2f}'.format(percent*2),'/',100,
                      end=end_char)

            # Print metrics
            metrics = self.get_metrics(x_train, y_train, cv)
            metrics_str = ''
            for key, value in metrics.items():
                metrics_str += '\t'+ key +': ' + '{:.4f}'.format(value['train']) + ' '
                self.metrics_list[key] = self.metrics_list.get(key, dict())
                for k, v in value.items():
                    self.metrics_list[key][k] = self.metrics_list[key].get(k, []) + [v]
            print(metrics_str)

            # Update learning rate
            self.learning_rate *= self.lr_decay

        # Plot metrics
        n_metrics = len(self.metrics_list.keys())
        
        for idx, key in enumerate(self.metrics_list):
            steps = np.arange(len(self.metrics_list[key]['train']))
            ax = plt.subplot(n_metrics, 1, idx+1)
            
            for set_name, values in self.metrics_list[key].items():
                ax.plot(steps, np.array(values), label = set_name)
                
            ax.set_title(key, y=0.85)
            ax.grid(True)
            ax.legend()

        plt.xlabel('epoch')
        plt.show()

    # Predict - forward pass through each layer
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    # Set loss function
    def get_loss_fn(self, loss):
        if loss.lower() == 'mean_squared_error':
            return losses.mse
        elif loss.lower() == 'categorical_cross_entropy':
            return losses.softmax_loss
        else :
            raise Exception('\'' + str(loss) + '\' loss function not found!')

    # Get metrics
    def get_metrics(self, x_train, y_train, cv = None):
        
        pred_train = self.predict(x_train)
        delta_train = pred_train-y_train
        if cv is not None:
            x_cv, y_cv = cv
            pred_cv = self.predict(x_cv)
            delta_cv = pred_cv - y_cv
            
        metrics = {}
        
        if self.loss_fn.__name__ == 'mse':
            mse_train = np.mean(np.sum(np.square(delta_train), axis=0,keepdims=True))
            metrics['loss'] = {'train': mse_train}
            if cv is not None:
                metrics['loss']['cv'] = np.mean(np.sum(np.square(delta_cv), axis=0,keepdims=True))

        elif self.loss_fn.__name__ == 'softmax_loss':
            # For numerical stability of log calculation, near-zero values are brought up to a small value - epsilon
            epsilon = 1e-10
            log_train = np.log(np.maximum(pred_train, epsilon))
            loss_train = -np.mean(np.sum(log_train * y_train, axis=0,keepdims=True))
            acc_train = (np.argmax(pred_train, axis=0) == np.argmax(y_train, axis=0)).mean()

            metrics['loss'] = {'train': loss_train}
            metrics['accuracy'] = {'train': acc_train}
            
            if cv is not None:
                log_cv = np.log(np.maximum(pred_cv, epsilon))
                loss_cv = -np.mean(np.sum(log_cv * y_cv, axis=0,keepdims=True))
                acc_cv = (np.argmax(pred_cv, axis=0) == np.argmax(y_cv, axis=0)).mean()

                metrics['loss']['cv'] = loss_cv
                metrics['accuracy']['cv'] = acc_cv

        else:
            raise Exception(f"Loss function '{self.loss_fn.__name__}' not found!")

        return metrics
