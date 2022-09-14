import torch

# Computations of gradient using the autograd package in python
# Gradient is very useful for the modal optimizations

d = torch.ones(6, requires_grad=True)
print(d)
# the requires_grad is by-default false
# What the requires_grad does is that, this will tell pytorch that
# it will need to calculate the gradients for this tensor later in the optimization steps
# So this means whenever we have variable in our model that we want to optimize then we need the gradients

x = torch.randn(3, requires_grad=True) #tensor with three random values
print(x)

y = x/2
print(y)
z = y*y*2
print(z)
p = x.mean()
print(p)

# Now calculating the backward of the gradient
p.backward() # dp/dx
print(x.grad)

# Now, we can eliminated tracking gradients using the three functions
# x.requires_grad_(false)
# print(x)

q = x.detach() # what it will do is, it creates a new vector with a tensor with same values but it does not require a gradient