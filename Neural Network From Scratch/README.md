# Neural Network From Scratch

## Description:

This is a Python implementation of a neural network package developed from scratch, inspired by the deep learning course by Andrew Ng. The fundamentals of neural networks: forward and backward propagation, activation functions, loss functions, and gradient descent are implemented with matrix algebra using NumPy.

## Usage:  

- Clone the repository using:  
`git clone https://github.com/eranda-ihalagedara/machine-learning-projects.git`  
or  
download the repository: [machine-learning-projects](https://github.com/eranda-ihalagedara/machine-learning-projects)

- Copy the folder `NN` under `./Neural Network From Scratch/` to your working directory.
- Import the package, build your model, and train with your data as in following code snippet:
```
import NN as nn

model = nn.Model([
    nn.Fully_Connected(32, 'relu', input_size=784),
    nn.Fully_Connected(64, 'relu'),
    nn.Fully_Connected(64, 'relu'),
    nn.Softmax(10, 'linear')
], learning_rate=0.01, lr_decay=0.995)

model.train(X_train,y_train, epochs=50)

model.predict(X_test)
```
**Important: Before building a model, make sure the input(X_train) and labels(y_train) are `numpy.ndarray`s of shapes (n,m) and (k,m) where `m` is the number of records or samples in the training set. In other words, features should be in rows and each sample should be in columns. Reshape your data accordingly, otherwise this would not provide intended results**

## Example  
You can view an example Jupyter Notebook [here]() where it goes through building a neural network model and train on the MNIST dataset.

## References

1. [Neural Networks and Deep Learning](https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&ab_channel=DeepLearningAI) by Andrew Ng  
2. [Improving Deep Neural Networks](https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&ab_channel=DeepLearningAI) by Andrew Ng  
3. [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1) by Thomas Kurbiel
4. [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) by  Eli Bendersky



## Disclaimer:  
This project is intended for educational purposes, and its primary goal is to help individuals understand the principles behind neural networks. For production-grade implementations, consider established deep learning frameworks like TensorFlow or PyTorch.
