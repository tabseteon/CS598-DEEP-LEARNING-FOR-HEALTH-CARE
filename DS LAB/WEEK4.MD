# Mortality Prediction With RNN
### To deal with sequential data, the most commonly used architecture is actually recurrent neural network (RNN)

## $$H_t = \phi(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$$
#### From the relationship between hidden variables $$H_t$$ and $$H_{t-1}$$ of adjacent time steps, we know that these variables captured and retained the sequence’s historical information up to their current time step, just like the state or memory of the neural network’s current time step. Therefore, such a hidden variable is called a hidden state. Since the hidden state uses the same definition of the previous time step in the current time step, the computation of $$H_t$$ is recurrent. Hence, neural networks with hidden states based on recurrent computation are named recurrent neural networks. Layers that perform the computation of $$H_t$$ in RNNs are called recurrent layers.
