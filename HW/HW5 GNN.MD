# Graph Neural Network (GNN)
### We will use PyTorch Geometric, which is a geometric deep learning extension library for PyTorch.
### We will use a simple Graph Convolution Network (GCN) to do enzyme multi-class classification (6 EC top-level classes)
### GCN (Graph Convolution Network): A deep learning model that learns directly from graph data (nodes and edges) by gathering information from neighboring nodes (aggregation).

### The GCN will have the following steps:

- Embed each node by performing multiple rounds of message passing
- Aggregate node embeddings into a unified graph embedding (readout layer
- Train the final classifier on the graph embedding

## 1 Build GCN Model Layer
```python
import math

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        # Initialize parameters.
        stdv = 1. / math.sqrt(out_channels)
        self.theta.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index):

        num_nodes = x.size(0)
        
        #1 adjacency matrix
        loop_edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        A_hat = torch.zeros((num_nodes, num_nodes), device=x.device)
        A_hat[loop_edge_index[0], loop_edge_index[1]] = 1.0
        
        #2 diagonal degree matrix
        deg = A_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.  # 0으로 나누기 방지
        D_hat_inv_sqrt = torch.diag(deg_inv_sqrt)
        norm_A = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        
        #3 the output X
        support = torch.mm(x, self.theta)
        ret = torch.mm(norm_A, support)
        
        return ret
```

## 2 Build GCN Model
```python
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self):

        super(GCN, self).__init__()
        
        # 1 Define the first convolution layer using `GCNConv()`. Set `out_channels` to 64
        self.conv1 = GCNConv(7, 64)
        # 2 Define the first activation layer using `nn.ReLU()`
        self.relu1 = nn.ReLU()
        # 3 Define the second convolution layer using `GCNConv()`. Set `out_channels` to 64
        self.conv2 = GCNConv(64, 64)
        # 4 Define the second activation layer using `nn.ReLU()`
        self.relu2 = nn.ReLU()
        # 5 Define the third convolution layer using `GCNConv()`. Set `out_channels` to 64
        self.conv3 = GCNConv(64, 64)
        # 6 Define the dropout layer using `nn.Dropout()`
        self.dropout = nn.Dropout()
        # 7 Define the linear layer using `nn.Linear()`. Set `output_size` to 2
        self.linear = nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
 
        # 1 Pass the data through the frst convolution layer
        x = self.conv1(x, edge_index)
        # 2 Pass the data through the first activation layer
        x = self.relu1(x)
        # 3 Pass the data through the second convolution layer
        x = self.conv2(x, edge_index)
        # 4 Pass the data through the second activation layer
        x = self.relu2(x)
        # 5 Pass the data through the third convolution layer
        x = self.conv3(x, edge_index)
        # 6 Obtain the graph embeddings using the readout layer with `global_mean_pool()`
        x = global_mean_pool(x, batch)
        # 7 Pass the graph embeddgins through the dropout layer
        x = self.dropout(x)
        # 8  Pass the graph embeddings through the linear layer
        x = self.linear(x)
        
        # 확률값(probabilities) 형태로 반환하기 위해 softmax 적용
        probs = torch.softmax(x, dim=1)
        
        return probs
        
GCN()
```

## 3 Training and Inferencing
```python
gcn = GCN()

# optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
# loss
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    gcn.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        
        # 1. 이전 단계에서 계산된 기울기(gradient) 초기화
        optimizer.zero_grad()
        
        # 2. 순전파(Forward pass): 모델에 데이터를 넣어 예측값 계산
        out = gcn(data.x, data.edge_index, data.batch)
        
        # 3. 손실(Loss) 계산
        loss = criterion(out, data.y)
        
        # 4. 역전파(Backward pass): 손실에 대한 기울기 계산
        loss.backward()
        
        # 5. 옵티마이저를 통해 가중치 업데이트
        optimizer.step()

def test(loader):
    gcn.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = gcn(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(200):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
```
