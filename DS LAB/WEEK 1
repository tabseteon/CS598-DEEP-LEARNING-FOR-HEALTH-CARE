## 1 Tensors
#### Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
#### Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. Tensors are also optimized for automatic differentiation

## 2 Loss
#### Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. 
#### To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value.

## 3 Autograd
#### When training the model, we want to find parameters that minimize the total loss across all training examples.
#### To do this, we will iteratively reduce the error by updating the parameters in the direction that incrementally lowers the loss function. This algorithm is called gradient descent. 
#### The most naive application of gradient descent consists of taking the derivative of the loss function.

##### Derivative: It measures how much the output changes when the input changes by a tiny amount.
##### Chain Rule If x → u → y (y depends on u, u depends on x):
##### Even a complex function can be broken into small steps, each differentiated separately, then multiplied together.
##### Forward pass: compute from input x to output y, while PyTorch records every operation as a graph
##### Backward pass: starting from y, walk backward toward x, applying the chain rule at each step to compute gradients automatically

##### x = torch.arange(4.0)
##### x.requires_grad_(True) # 	Declares that gradients should be tracked for this variable
##### y = 2 * torch.dot(x, x)  # y = 2x^T x = 28
##### y.backward() # Stores dy/dx in x.grad
##### x.grad # Returns the stored gradient

##### PyTorch uses its own data type called torch.Tensor
##### matplotlib (e.g. plt.plot) is designed to take numpy arrays or lists as input
##### If the tensor has requires_grad=True, it's attached to a computational graph (gradient tracking), so calling .numpy() directly throws an error
##### .detach() means: "remove this tensor from the computational graph and give me a plain tensor with just the values, no longer tracking gradients"
##### After that, you can safely call .numpy() to convert it into a numpy array
