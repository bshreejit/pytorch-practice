import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd

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
    data = pd.read_csv('./linear_regression/real_estate/data/real-estate.csv', header=None)
    df = data.iloc[1: , 1:]
    inputs = df.loc[:, :6]
    targets = df.iloc[: , -1]

    inputs = torch.from_numpy(inputs.to_numpy().astype(np.float32))
    targets = torch.from_numpy(targets.to_numpy().astype(np.float32))
    targets = targets.reshape([414, 1])

    batch_size = 20
    train_ds = TensorDataset(inputs, targets)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)

    model = nn.Linear(6, 1) # nn.Linear(in-features, out-features)

    # Generating prediction
    prediction = model(inputs)

    loss_fn = F.mse_loss 
    loss = loss_fn(prediction, targets)
    print(loss)

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    fit(100, model, loss_fn, opt, train_dl)
    print("Final prediction :: ", model(inputs))

if __name__ == "__main__":
    main()

