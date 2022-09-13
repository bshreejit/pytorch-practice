import torch
import numpy as np

# @@@@ INTEROPERABILITY OF TENSORS AND NUMPY
# To leverage the existing numpy ecosystem of tools and libraries like pandas for i/o and data analysis,
# matplotlib for plotting and visualization, and open-cv for image and video processing

# creating a tensor
a = torch.ones(5)
print("Tensor array A :: ", a)
print("The type of the tensor array A ::", type(a))

# creating a numpy array
# changing from the tensor to numpy array
b = a.numpy()
print("Prints the numpy array :: ", b)
print("Prints the type of the numpy array :: ", type(b))

# Things to be careful while using the numpy and tensor array with each other
# -> if the tensors is on the cpu and not the gpu, then both the 
# -> objects in this case will share the same memory location
# -> This means if we change one, we will be eventually then changing the value of other 

# -> In this case if we modify a, then b will be automatically modified
a.add_(1)
print("After adding 1 to a :: ", a)
print("We can see the change occurring in b automatically:: ", b)

# Now doing vice-versa i.e., from numpy to tensor
p = np.ones(5)
print("Numpy array p :: ", p)
q = torch.from_numpy(p)
print("Tensor q :: ", q) # here by-default the data type is set as float 64

# Also
p += 1
print("Updated p :: ", p)
print("Updated q, occurring automatically :: ", q)

# Now to avoid or to solve the above case, we can do the following by checking the availability of the GPU
if torch.cuda.is_available(): # returns false in my working env, i.e, mac
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # This will create a tensor and put it on the GPU
    
    # Or, alternatively we create the tensor first
    y = torch.ones(5)
    # And then only we move it to the device to the GPU
    y = y.to(device)

    # Now if we perform
    z = x + y #It will be performed on the GPU and it might be much faster

    # Now, if we run the below code
    z.numpy()
    # This will throw an error because the numpy can only handle CPU tensor, so we cannot convert a GPU tensor back to numpy

    #  But, we can do it by moving it back to the CPU
    z = z.to("cpu") # So, now it will be on the CPU again



