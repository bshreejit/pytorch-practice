import numpy as np
import torch

def model(input, w, b):
    return input @ w.t() + b

def calculateMSE(prediction, target):
    difference = target - prediction
    return torch.sum(difference*difference)/difference.numel()

def main():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43], 
                    [91, 88, 64], 
                    [87, 134, 58], 
                    [102, 43, 37], 
                    [69, 96, 70]], dtype='float32')
    # Targets (apples, oranges) 
    targets = np.array([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119]], dtype='float32')

    # Converting inputs and targets to tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    epoch = 100

    # Initializing the weights and biases
    weights = torch.randn(2, 3, requires_grad=True)
    biases = torch.randn(2, requires_grad=True)

    # TRAINING A MODEL For multiple Epochs
    for i in range(epoch):
        prediction = model(inputs, weights, biases)

        # Calculating the loss i.e, mean squared mean (MSE)
        loss = calculateMSE(targets, prediction)
        # print("Initial loss :: ", loss)

        # Computing gradients
        loss.backward()

        # The gradients are stored in the `.grad` property of the respective tensors
        # print(weights.grad)
        # print(biases.grad)

        # Now, we update the weights and biases using the gradients computed above
        # Adjusting weights & reset gradients
        with torch.no_grad():
            # no_grad() -> indicates no modification, calculation and tracking should be done while updating the weights and biases
            weights -= weights.grad * 1e-5 # 1e-5 is the learning rate. The L.R is such small because we dont want to modify the weights by very large amounts
            biases -= biases.grad * 1e-5
            weights.grad.zero_()
            biases.grad.zero_()
            # we reset the gradients to zero by invoking the .zero_() method, otherwise in next iteration, value gets added to the new loss.backward() 
    
    print("Loss/MSE after 100 epoch :: ", loss)
    print("Target :: ", targets)
    print("Predicted O/P after 100 epoch :: ", prediction)

if __name__ == "__main__":
    main()


