# Recurent Neural Network

### We build bi-directional RNN on diagnosis codes to perform Heart Failure Prediction. The recurrent nature of RNN can model the temporal relation of different visits of patients.

## 1 Build DataSet
## 1-A Split DataSet
```python
from torch.utils.data.dataset import random_split

split = int(len(dataset)*0.8)

lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)
```

## 1-B DataLoader
```python
from torch.utils.data import DataLoader

def load_data(train_dataset, val_dataset, collate_fn):

    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)
```

## 2 Naive RNN
1. Pass the sequence through the embedding layer
2. Sum the embeddings for each diagnosis code up for a visit of a patient
3. Pass the embeddings through the RNN layer
4. Obtain the hidden state at the last visit
5. Do 1-4 for both directions and concatenate the hidden states (히든 스테이트 정보 이어붙이기)
6. Pass the hidden state through the linear and activation layers


```python
# Build and Load Naive RNN
class NaiveRNN(nn.Module):

    def __init__(self, num_codes):

        super().__init__()

        self.embedding = nn.Embedding(num_codes, embDimSize)
        self.rnn = nn.GRU(input_size=embDimSize, hidden_size=hidden_size, batch_first=True)
        self.rev_rnn = nn.GRU(input_size=embDimSize, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):

        # 1. Pass the sequence through the embedding layer
        x = self.embedding(x)
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        x = sum_embeddings_with_mask(x, masks)
        # 3. Pass the embeggings through the RNN layer;
        output, _ = self.rnn(x)
        # 4. Obtain the hidden state at the last visit.
        true_h_n = get_last_visit(output, masks)
        # 5. Do the step 1-4 again for the reverse order (rev_x)
        #    and concatenate the hidden states for both directions
        rev_x = self.embedding(rev_x)
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        rev_output, _ = self.rev_rnn(rev_x)
        true_h_n_rev = get_last_visit(rev_output, rev_masks)
        # 6. Pass the hidden state through the linear and activation layers.
        logits = self.fc(torch.cat([true_h_n, true_h_n_rev], 1))        
        probs = self.sigmoid(logits)
        return probs.view(batch_size)

# load the model
naive_rnn = NaiveRNN(num_codes = len(types))
```

## 2 Model Training and Evaluation

## 2-A Loss and Optimizer
```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(naive_rnn.parameters(), lr=0.001)
```

## 2-B Evaluation
```python
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def eval_model(model, val_loader):

    model.eval() # switch to evaluation mode

    y_pred = torch.LongTensor()   # predicted classes (0 or 1)
    y_score = torch.Tensor()      # raw predicted probabilities
    y_true = torch.LongTensor()   # ground-truth labels

    for x, masks, rev_x, rev_masks, y in val_loader:
        y_hat = model(x, masks, rev_x, rev_masks) # predicted probability from the model
        y_score = torch.cat((y_score,  y_hat.detach().to('cpu')), dim=0) # accumulate raw probabilities (needed for ROC-AUC)
        y_hat = (y_hat > 0.5).int() # convert probability to binary prediction
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0) # accumulate binary predictions
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0) # accumulate ground-truth labels
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc
```

## 2-C Training and Evaluation
```python
def train(model, train_loader, val_loader, n_epochs):

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:

            # 1. zero grad
            optimizer.zero_grad()
            # 2. model forward
            y_hat = model(x, masks, rev_x, rev_masks)
            # 3. calculate loss
            loss = criterion(y_hat, y)
            # 4. loss backward
            loss.backward()
            # 5. optimizer step
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

```
