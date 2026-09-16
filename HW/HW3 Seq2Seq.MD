# The Sequence-to-Sequence (Seq2seq) Autoencoder Model
### We use the sequence-to-sequence (Seq2seq) autoencoder model to generate patient EHR embedding and use these embeddings to conduct unsupervised patient clustering.

## 1 Build the Seq2Seq model
### In this HW, we will build a naive Seq2Seq autoencoder model which uses a CNN with attention mechanism as encoder and a GRU as decoder. The input to the encoder will be the multi-hot code vector at each timestep. The decoder will try to reconstruct the input at each timestep based on the hidden embedding of the encoder network. In this way, we can store the longitudinal patient EHR data in a vector (i.e., the hidden state of the encoder network).

## 1-A Build the encoder model
```python
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 105), padding=(1, 0))
        self.att = nn.Linear(128, 1)

    def forward(self, x, masks):
        # 1. add channel dimension, then pass through CNN
        x = x.unsqueeze(1)                     # (batch_size, 1, # visits, # diagnosis codes)
        hidden_states = self.cnn(x)            # (batch_size, 128, # visits, 1)
        
        # 2. reshape to (batch_size, # visits, 128)
        hidden_states = hidden_states.squeeze(-1)      # (batch_size, 128, # visits)
        hidden_states = hidden_states.permute(0, 2, 1)  # (batch_size, # visits, 128)
        
        # 3. calculate attention score
        att_score = self.att(hidden_states)     # (batch_size, # visits, 1)
        att_score = att_score.squeeze(-1)       # (batch_size, # visits)
        
        # 4. mask out padded visits
        att_score = att_score.masked_fill(~masks, -1e9)
        
        # 5. softmax to get attention weights
        att_weight = torch.softmax(att_score, dim=1)   # (batch_size, # visits)
        
        # 6. weighted sum to get context vector
        states = torch.sum(att_weight.unsqueeze(-1) * hidden_states, dim=1)  # (batch_size, 128)
        
        return states
```

## 1-B Build the decoder model
```python
class Decoder(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.rnn = nn.GRU(input_size=105, hidden_size=128, batch_first=True)

        # MLP head to reconstruct the diagnosis vector from the hidden state      
        self.linear1 = nn.Linear(128, 128)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(128, 105)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hiddens):

        # Unsqueeze the input at dimension 1 -> (batch_size, 1, # diagnosis codes)
        x = x.unsqueeze(1)
        # Unsqueeze the hiddens at dimension 0; -> (1, batch_size, 128)
        hiddens = hiddens.unsqueeze(0)

        # run one GRU step, using `hiddens` as the initial hidden state
        _, h_n = self.rnn(x, hiddens)
        h_n = h_n.squeeze(0)

        # reconstruct the diagnosis vector from the new hidden state
        preds = self.linear1(h_n)
        preds = self.act(preds)
        preds = self.linear2(preds)
        preds = self.sigmoid(preds)
        return preds, h_n

```

## 1-C Connect the encoder and decoder
### 1 Encode: Pass the full visit sequence through the Encoder to obtain a summary vector (hidden).
### 2 Start decoding: Feed a zero vector as the initial input and hidden as the initial state into the Decoder to predict the first visit.
### 3 Loop
###   - Use random.random() to decide whether to apply teacher forcing.
###   - If teacher forcing, use the ground truth as the next input. Otherwise, use the model's previous prediction.
###   - Feed the chosen input and the previous hidden state into the Decoder to get a new prediction and hidden state, and store the prediction.
### 4 Collect results: Stack all predictions with torch.stack and reshape
```python
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, teacher_forcing_ratio):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def forward(self, x, masks):
        preds = []
        
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # step 1
        hidden = self.encoder(x, masks)
        
        # step 2
        x_decode = torch.zeros(batch_size, 105)
        cur_pred, h = self.decoder(x_decode, hidden)
        preds.append(cur_pred)
        
        for t in range(1, seq_len):
            # step 3.1
            teacher_force = random.random() < self.teacher_forcing_ratio
            # step 3.2
            x_decode = x[:, t-1, :] if teacher_force else cur_pred
            cur_pred, h = self.decoder(x_decode, h)
            
            # step 3.3
            preds.append(cur_pred)
        
        # step 4
        return torch.stack(preds).permute(1,0,2)
```
