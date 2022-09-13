import numpy as np
import torch

# Step by step analysis of linear-regression

def model(input, w, b):
    return input @ w.t() + b
    # here, @ -> matrix multiplication, .t() -> transpose of a matrix

def calculateMSE(prediction, target):
    difference = target - prediction
    return torch.sum(difference*difference)/difference.numel()
    # here, torch.sum returns the sum of all the resulting elements of the tensor whereas the numel() returns the number of elements in the tensor

def main():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43], 
                    [91, 88, 64], 
                    [87, 134, 58], 
                    [102, 43, 37], 
                    [69, 96, 70]], dtype='float32')
    # Targets (apples, oranges) i.e, yield of the apples and the oranges on certain regions based on the three parameters i.e, temp, rainfall and humidity
    targets = np.array([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119]], dtype='float32')

    # Converting inputs and targets to tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    print("I/P's", inputs)
    print("Targets", targets)

    # In a linear regression model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias
    # Initializing the weights and biases
    weights = torch.randn(2, 3, requires_grad=True)
    biases = torch.randn(2, requires_grad=True)
    # torch.randn creates a tensor with the given shape, with elements picked randomly from a normal distribution with mean 0 and standard deviation 1.
    print('Weights :: ', weights)
    print('Biases :: ', biases)

    # TRAINING A MODEL
    initial_prediction = model(inputs, weights, biases)
    print("Initial Prediction :: ", initial_prediction)

    # Calculating the loss i.e, mean squared mean (MSE)
    initial_loss = calculateMSE(targets, initial_prediction)
    print("Initial loss :: ", initial_loss)

if __name__ == "__main__":
    main()


