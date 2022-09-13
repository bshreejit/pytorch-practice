import torch

# Creating tensors
x = torch.tensor(1.)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
print(x, w, b)
# In the above tensors, w and b have the extra parameter called "required_grad" which is set to 'true'
# By default the "required_grad" is set to false if nothing is specified


# Performing arithmetic operations in the above tensors
y = w * x + b
print("Output :: ", y)

# AUTOGRAD ->> Automatic Gradient
# Now, pytorch is unique because it can automatically compute the derivative of "Y" wrt the tensors that has 'required_grad' parameter set as true, i,e. w and b
y.backward() 
# backward() -> is used to compute the gradient during the backward pass in a neural network. The gradients are computed when this method is executed. 
# This is quite useful because derivatives are very important for the optimization algorithm i.e, gradient descent

# These gradients/derivatives are stored in the respective tensors which can be acquired by .grad property of that respective tensor
print('dy/dx:', x.grad) # displays none, because the required_grad is initially set to false
print('dy/dw:', w.grad)
print('dy/db:', b.grad)


# NOTE
# Each gradient calculation can slow down the model significantly, so for large models which may have large parameters, we only calculate the gradients(derivatives) when we only need them