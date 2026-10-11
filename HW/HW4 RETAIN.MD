# RETAIN: Interpretable Heart Failure Prediction from Diagnosis Codes
### PyTorch implementation of RETAIN (Choi et al., 2016), a recurrent model with two-level reverse-time attention, applied to heart failure (HF) prediction
### RETAIN predicts whether the patient has heart failure, and its attention weights show which visits and which features drove the prediction.
### Paper: RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism

## 1 Build the dataset 
### 1-A Collate Function
#### we will convert the data to tensor using a collate function. 
#### This collate function will be called by DataLoader after fetching a list of samples using the indices from CustomDataset to collate the list of samples into batches.

```python
def collate_fn(data):
    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)
    
    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)
    
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            """
            TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
            """
            
            n_codes = len(visit)
            
            # RETAIN 모델은 정방향 RNN과 역방향 RNN을 각각 돌려서 결과를 결합
            
            # 정방향
            x[i_patient, j_visit, :n_codes] = torch.tensor(visit, dtype=torch.long)
            masks[i_patient, j_visit, :n_codes] = True
        
            # 역방향: 방문 순서만 뒤집어서 배치
            rev_j = len(patient) - 1 - j_visit
            rev_x[i_patient, rev_j, :n_codes] = torch.tensor(visit, dtype=torch.long)
            rev_masks[i_patient, rev_j, :n_codes] = True
    
    return x, masks, rev_x, rev_masks, y
```

### 1-B DataLoader

```python
from torch.utils.data import DataLoader

def load_data(train_dataset, val_dataset, collate_fn):
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader


train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)
```

## 2 RETAIN
## ATTENTION MECHANISM : 입력의 여러 부분 중 지금 필요한 곳에 더 집중해서 가중평균을 내는 방법
#### When generating a translation of a source text, 
#### we first pass the source text through an encoder to obtain a sequence of encoder hidden states
#### Then, at decoding steps, we construct a context vector that is a weighted average of encoder hidden states
#### The weights are computed from the current decoder hidden state and the encoder hidden states and normalized to form a categorical probability distribution

### Two different attention mechanisms here
- attention 𝛼 : a scalar for the i-th visit, tells you the importance of the i-th visit.
- attention 𝛽 : a vector, tells us more detailed view of what are the important features within a visit.


### Unfolded view of RETAIN’s architecture
#### Step 1: Embedding
#### Step 2: generating 𝛼 values using RNN-𝛼
#### Step 3: generating 𝛽 values using RNN-𝛽
#### Step 4: Generating the context vector using attention and representation vectors
#### Step 5: Making prediction

## 2-A AlphaAttention
```python
class AlphaAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.a_att = nn.Linear(embedding_dim, 1)

    def forward(self, g, rev_masks):
        # 1 Calculate the attention score
        score = self.a_att(g) # 1. Calculate the attention score
        # 2 Mask out the padded visits in the attention score with -1e9
        visit_mask = (torch.sum(rev_masks, dim=-1) > 0).unsqueeze(-1)
        score = score.masked_fill(~visit_mask, -1e9)
        # 3 Perform softmax on the attention score to get the attention value
        alpha = torch.softmax(score, dim=1)
```

## 2-B BetaAttention
```python
class BetaAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.b_att = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, h):

        # scale from -1 to 1
        beta = torch.tanh(self.b_att(h))
        return beta
```

## 2-C Attention Sum

```python
def attention_sum(alpha, beta, rev_v, rev_masks):
    
    # Mask select the hidden states for true visits (not padding visits)
    visit_mask = (torch.sum(rev_masks, dim=-1) > 0).unsqueeze(-1).float()
    weighted = alpha * beta * rev_v

    # Sum them up
    c = torch.sum(weighted * visit_mask, dim=1)

    return c
```

## 2-D Build RETAIN

```python
class RETAIN(nn.Module):
    
    def __init__(self, num_codes, embedding_dim=128):
        super().__init__()
        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(embedding_dim)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(embedding_dim)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(embedding_dim, 1)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, masks, rev_x, rev_masks):

        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embegginds through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        alpha = self.att_a(g, rev_masks)
        beta = self.att_b(h)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze(dim=-1)
    
# load the model here
retain = RETAIN(num_codes = len(types))
retain
```

## 3 Training and Inferencing

## 4 Sensitivity analysis
