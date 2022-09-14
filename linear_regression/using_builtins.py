# Since in deep-learning, the steps performed in initial_mse.py and gradient_descent.py are common patterns
# So, to ease coding all the way from scratch, pyTorch provides several libraries and classes that can do this same thing very easily

import torch
import numpy as np
# torch.nn contains utility classes for building neural networks
import torch.nn as nn
import torch.nn.functional as F
# TensorDataset allows to access rows from inputs and targets as tuples
from torch.utils.data import TensorDataset
# DataLoader splits the data into batches of a predefined size while training, also helps in shuffling and random sampling of the data
from torch.utils.data import DataLoader

def fit(epochs, model, loss_fn, opt, train_dl):
    for epoch in range(epochs):
        # Train with batches of data
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            # Updating parameters using gradients
            opt.step()
            # Resetting the gradients to zero
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

def main():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43], 
                    [91, 88, 64], 
                    [87, 134, 58], 
                    [102, 43, 37], 
                    [69, 96, 70], 
                    [74, 66, 43], 
                    [91, 87, 65], 
                    [88, 134, 59], 
                    [101, 44, 37], 
                    [68, 96, 71], 
                    [73, 66, 44], 
                    [92, 87, 64], 
                    [87, 135, 57], 
                    [103, 43, 36], 
                    [68, 97, 70]], 
                    dtype='float32')

    # Targets (apples, oranges)
    targets = np.array([[56, 70], 
                        [81, 101], 
                        [119, 133], 
                        [22, 37], 
                        [103, 119],
                        [57, 69], 
                        [80, 102], 
                        [118, 132], 
                        [21, 38], 
                        [104, 118], 
                        [57, 69], 
                        [82, 100], 
                        [118, 134], 
                        [20, 38], 
                        [102, 120]], 
                    dtype='float32')

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    batch_size = 4
    train_ds = TensorDataset(inputs, targets)
    # train_ds[0:3]
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    # Shuffling basically helps in randomizing the input of the optimization algorithm, which leads to a faster reduction in the loss

    # Now we can use the DataLoader in a loop, it prints one batch of data with the given batch size after shuffling 
    # for xb, yb in train_dl:
    #     print(xb)
    #     print(yb)
    #     break

    # nn.Linear is the class which can help define the model automatically
    model = nn.Linear(3, 2) # nn.Linear(in-features, out-features)

    # Generating prediction
    prediction = model(inputs)

    # To calculate loss function, a built-in loss function mse_loss can be used
    loss_fn = F.mse_loss #defining a loss function
    loss = loss_fn(prediction, targets)

    # Now, modifying the weights and biases is also made simple using the optimizer optim.SGD
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    # model.parameters() lets the optimizer know which of the matrices should be modified during the update step

    fit(100, model, loss_fn, opt, train_dl)
    print("Final prediction :: ", model(inputs))

if __name__ == "__main__":
    main()

