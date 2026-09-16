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
#### - Use random.random() to decide whether to apply teacher forcing.
#### - If teacher forcing, use the ground truth as the next input. Otherwise, use the model's previous prediction.
#### - Feed the chosen input and the previous hidden state into the Decoder to get a new prediction and hidden state, and store the prediction.
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
## 4 Model Training and Evaluation
## 4-A Loss and optimizer
```python
criterion = nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-2)
```

## 4-B Evaluation
```python
from sklearn.metrics import f1_score, accuracy_score

def eval(model, val_loader):

    model.eval()

    x_pred = []
    x_true = []
    for x, masks, y in val_loader:
        x_hat = model(x, masks).detach().numpy()
        x_hat = np.round(x_hat)
        for i in range(len(x_hat)):
            for j in range(len(x_hat[i])):
                if masks[i, j] == 1:
                    x_pred.append(x_hat[i,j])
                    x_true.append(x.numpy()[i,j])
    
    f = f1_score(y_pred=x_pred, y_true=x_true, average='micro')
    acc = accuracy_score(y_pred=x_pred, y_true=x_true)
    return f, acc
```
## 4-C Training and Evaluation

```python
def train(model, train_loader, val_loader, n_epochs):

    for epoch in range(n_epochs):
        seq2seq.train()
        train_loss = 0
        for x, masks, y in train_loader:
            optimizer.zero_grad()
            x_hat = seq2seq(x, masks)
            loss = criterion(x_hat, x)
            loss = loss * masks.float().unsqueeze(-1)
            loss = torch.mean(torch.sum(loss, dim=[1,2]))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        f, acc = eval(model, val_loader)
        print('Epoch: %d \t Validation f: %.2f, acc: %.2f'%(epoch+1, f, acc))

    
# number of epochs to train the model
n_epochs = 100

train(seq2seq, train_loader, val_loader, n_epochs)
```

## 5 Unsupervised patient clustering
### 5-A Generate patient embeddings
```python
# Use trained encoder model to get patient embeddings
def get_embeddings(model, data_loader):

    seq2seq.eval()
    embeddings = []
    labels = []

    for x, masks, y in data_loader:
        # use only the Encoder to generate patient embeddings
        cur_embd = seq2seq.encoder(x, masks) 
        # collect embeddings and labels, batch by batch
        embeddings += list(cur_embd.detach().numpy())
        labels += list(y.numpy())
        
    return np.array(embeddings), np.array(labels)

    
embeddings, labels = get_embeddings(seq2seq, val_loader)
```

### 5-B Cluster these embeddings
```python
# 1 Use K-means to generate patient clusters. (K=2)
# 2 Use calinski harabaz score to evaluate clustering results

from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans

def get_clusters(embeddings):

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    score = calinski_harabasz_score(embeddings, labels)
    
    return labels
    
cluster_labels = get_clusters(embeddings)
```

### 5-C Visualize the embeddings and clustering results
```python
from sklearn import manifold
import matplotlib.pyplot as plt

tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
tsne_embd = tsne.fit_transform(embeddings)

plt.scatter(tsne_embd[:,0], tsne_embd[:,1], c=labels)
plt.show()
```
