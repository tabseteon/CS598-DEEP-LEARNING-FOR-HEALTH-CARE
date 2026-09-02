# Neural Network on Heart Failure Prediction

### PyTorch is a framework for building and training neural networks. PyTorch takes tensors and makes it simple to move them to GPUs for the faster processing needed when training neural networks. It also provides a module that automatically calculates gradients (for backpropagation) and another module specifically for building neural networks.

## NN with PyTorch
### 1 Train/Test Data Split
### 2 Batch 
#### The data loader is created with a batch size of  32, and shuffle=True. The batch size is the number of samples the model processes at once before updating weights. Instead of updating weights after every single sample (too slow, unstable) or after the entire dataset at once (too much memory), we split the data into small groups. And shuffle=True tells it to shuffle the dataset every time we start going through the data loader again.

### 3 Build the Model
```python
class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(1473, 64) # Fully Connected Layer
        self.fc2 = nn.Linear(64, 32) # Fully Connected Layer
        self.dropout = nn.Dropout(p=0.5) # 학습 중에 뉴런의 50%를 랜덤하게 꺼버림(0으로 만듦) → 과적합(overfitting) 방지용
        self.fc3 = nn.Linear(32, 1) # Fully Connected Layer


    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# initialize the NN
model = Net()
print(model)
```

### 4 Train the Network
#### Neural networks with non-linear activations work like universal function approximators. Stacking only linear layers is still just linear function, no matter how many layers you add — a combination of linear functions is still linear.
Adding non-linear activation functions lets the network capture complex, non-linear patterns.
#### with a classification problem such as Heart Failure Prediction, we are using the Sigmoid function to predict heart failure probability.

```python
# With a Sigmoid output, you use binary cross-entropy loss (BCELoss).
criterion = nn.BCELoss()

# Optimizer can update the weights with the gradients.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
n_epochs = 100

train_loss_arr = []
for epoch in range(n_epochs):
    
    # prep model for training
    model.train()
    
    train_loss = 0
    for x, y in train_loader:
        """ Step 1. clear gradients """
        optimizer.zero_grad()
        """ Step 2. perform forward pass """
        y_hat = model(x)
        y_hat = y_hat.view(y_hat.shape[0]) # [batch_size, 1] → [batch_size] 
        """ Step 3. calculate the loss """
        loss = criterion(y_hat, y)
        """ Step 4. backward pass """
        loss.backward()
        """ Step 5. optimization """
        optimizer.step()
        """ Step 6. record loss """
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)

    if epoch % 20 == 0:
        train_loss_arr.append(np.mean(train_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        evaluate(model, val_loader)
```

#### To train the model, you should follow the following step:
- Clear the gradients of all optimized variables
- Forward pass: compute predicted outputs by passing inputs to the model
- Calculate the loss
- Backward pass: compute gradient of the loss with respect to model parameters
- Perform a single optimization step (parameter update)
- Update average training loss
