# pyTorch is a library for processing tensors. A tensor can be a number, matrix, vector or any n-dimensional array
import torch

# .shape in a tensor fives the length along each dimension. It is the property of the tensor

# Working with Numbers
t1 = torch.tensor(8.)
print("Tensor t1: ", t1) 
print("Tensor t1 shape: ", t1.shape)
# The 8. in the above tensor indicates the floating number

# the .dtype expresses the datatype the tensor t1 is
print("Tensor t1 data  type: ", t1.dtype) 

# Creating a Vector
t2 = torch.tensor([2., 34, 3, 6])
print("Tensor t2: ", t2) #In the above tensor t2, the first element in the vector is a floating number so the tensor automatically converts all the other vector numbers into a float
print("Tensor t2 shape: ", t1.shape)

# Creating a Matrix
t3 = torch.tensor([[4., 6], 
                   [7, 5], 
                   [9, 1]])
print("Tensor t3: ", t3)
print("Tensor t3 shape: ", t3.shape)
    
# Working with a 3-dimensional array
t4 = torch.tensor([
    [[1, 2, 3], 
     [3, 4, 5]], 
    [[5, 6, 7], 
     [7, 8, 9.]]])
print("Tensor t4: ", t4)
print("Tensor t4 shape: ", t4.shape)

# Empty Tensor
x = torch.empty(1)
# 2-D tensor with all 0's
y = torch.zeros(2,2)
# 2-D tensor with all 1's
z = torch.ones(2,2)
# 3-D tensor with all random variables
a = torch.rand(2,2,3)
print("Tensor a: ", a)