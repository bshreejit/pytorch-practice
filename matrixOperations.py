import torch

# Tensor p and q initialization
p = torch.rand(2, 2)
q = torch.rand(2, 2)

# Tensor (Matrix) addition
print("Adding tensor p and q :: ")
print("First way ::", p + q)
print("Second way ::", torch.add(p, q))
# Or, also can use an in-place addition as below
print("In-place addition :: ", q.add_(p)) # This in-place addition will modify the value of q

# NOTE
# In pytorch every function which has a trailing underscore
# like add_() will do an in-place operation, hence it will modify the
# variable that it is applied on

# Tensor (Matrix) subtraction
print("Subtracting tensor p and q :: ")
print("First way :: ", p - q)
print("Second way :: ", torch.sub(p, q))
# Or, an in-place subtraction as below
print("In-place subtraction:: ", q.sub_(p))

# Tensor (Matrix) multiplication
print("Multiplying tensor p and q :: ")
print("First way :: ", p * q)
print("Second way :: ", torch.mul(p, q))
# Or, an in-place multiplication as below
print("In-place multiplication:: ", q.mul_(p))

# We can also do a SLICING operation like in the numpy
x = torch.rand(5,3) # Represents 5 rows and 3 columns

print("Original x :: ", x)
print("Slicing operation x[:,0] :: ", x[:,0]) # this slicing operation prints all the rows but only of first column

print("Original x :: ", x)
print("Slicing operation x[1,:] :: ", x[1,:]) # this slicing operation prints only second row second but all the columns

print("Original x :: ",x)
print("Slicing operation x[1,1] :: ", x[1,1]) # here the slicing operation prints the (1,1) element as of in matrix
# To get the actual value of a single element in a tensor, we can use the item()
print("Actual value of [1,1], we use x[1,1].item() :: ", x[1,1].item())

# RESHAPING a tensor value

y = torch.rand(4,4)
print("Original y :: ", y)
print("Printing in one dimension view :: ", y.view(16)) # printing in one dimension

