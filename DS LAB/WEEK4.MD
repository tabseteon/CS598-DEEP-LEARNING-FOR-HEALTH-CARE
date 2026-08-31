# Mortality Prediction With RNN
### To deal with sequential data, the most commonly used architecture is actually recurrent neural network (RNN)

## $$H_t = \phi(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$$
#### From the relationship between hidden variables $$H_t$$ and $$H_{t-1}$$ of adjacent time steps, we know that these variables captured and retained the sequence’s historical information up to their current time step, just like the state or memory of the neural network’s current time step. Therefore, such a hidden variable is called a hidden state. Since the hidden state uses the same definition of the previous time step in the current time step, the computation of $$H_t$$ is recurrent. Hence, neural networks with hidden states based on recurrent computation are named recurrent neural networks. Layers that perform the computation of $$H_t$$ in RNNs are called recurrent layers.

## Modern RNN Models
#### We have introduced the basics of RNNs, which can better handle sequence data. However, such techniques may not be sufficient for practitioners when they face a wide range of sequence learning problems nowadays.
### 1 Gated Recurrent Units (GRU)
#### The key distinction between vanilla RNNs and GRUs is that the latter support gating of the hidden state. This means that we have dedicated mechanisms for when a hidden state should be updated and also when it should be reset. These mechanisms are learned and the network can selectively keep or forget information.

### 2 Long Short-Term Memory (LSTM)
#### LSTM introduces a memory cell (or cell for short) that has the same shape as the hidden state, engineered to record additional information. To control the memory cell we need a number of gates. One gate is needed to read out the entries from the cell. We will refer to this as the output gate. A second gate is needed to decide when to read data into the cell. We refer to this as the input gate. Last, we need a mechanism to reset the content of the cell, governed by a forget gate.

## Build NaiveRNN

```python

class NaiveRNN(nn.Module):
    
    """
    TODO: implement the naive RNN model above.
    """
    
    def __init__(self):
        super().__init__()
        """
        TODO: 
        1. Define the RNN using `nn.GRU()`
        2. Define the linear layers using `nn.Linear()` Set `output_size` to 1.
        3. Define the final activation layer using `nn.Sigmoid()
        """
        
        self.rnn = nn.GRU(input_size=TOTAL_NUM_CODES, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, length):
        """
        1. Pass the inputs through the RNN layer;
        2. Obtain the hidden state at the last visit.
        3. Pass the hidden state through the linear and activation layers.
        """        

        hidden_states, _ = self.rnn(x)
        last_hidden_state = get_last_visit(hidden_states, length)
        out = self.fc(last_hidden_state)
        probs = self.sigmoid(out)
        return probs.squeeze(-1)
    

# load the model here
model = NaiveRNN()
model
```

