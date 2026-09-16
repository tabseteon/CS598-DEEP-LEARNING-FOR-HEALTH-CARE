# Convolutional Neural Network
### We train a CNN model to classify Pneumonia disease (Pneumonia/Normal) based on chest X-Ray images.

## Build the Model
### We define a CNN architecture. Instead of an MLP, which used linear, fully-connected layers, you will use the following:

- Convolutional layers, stack of filtered images.
- Maxpooling layers, reducing the x-y size of an input, keeping only the most active pixels from the previous layer.
- The usual Linear + Dropout layers to avoid overfitting and produce 2-dim output.

### Typical CNN architicture consists of [INPUT - CONV - RELU - POOL - FC] layers.

## A Define Simple CNN
### The more convolutional layers you include, the more complex patterns the model can detect. For now, the final model includes 3 convolutional layers as well as linear layers + dropout in between to avoid overfitting.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers: 3 -> 16 -> 32 -> 64 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Feature map size after 3x (conv + pool): 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # binary classification: NORMAL / PNEUMONIA

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        
        x = self.pool(self.relu(self.conv1(x)))  # (B, 16, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 32, 56, 56)
        x = self.pool(self.relu(self.conv3(x)))  # (B, 64, 28, 28)
        
        x = x.view(x.size(0), -1)  # flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

## B Using Predefined CNN Model

```python
def get_cnn_model():
    
    from torchvision import models
    
    num_classes = 2
    
    # Load ResNet18 architecture without pretrained weights (loaded manually later)
    model = models.resnet18(pretrained=False)
    
    # Replace the final fully-connected layer to output 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze all layers except the new fc layer for computation efficiency
    for param in model.named_parameters():
        if param[0].split(".")[0] == 'fc':
            continue
        param[1].requires_grad = False
    
    return model

```

## C Train the Network
#### Pre-trained weights loaded due to the computation environment constraint
### Training Steps:
- Clear the gradients of all optimized variables
- Forward pass: compute predicted outputs by passing inputs to the model
- Calculate the loss
- Backward pass: compute gradient of the loss with respect to model parameters
- Perform a single optimization step (parameter update)
- Update average training loss

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

n_epochs = 1

def train_model(model, train_dataloader, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    import torch.optim as optim
    """
    :param model: A CNN model
    :param train_dataloader: the DataLoader of the training data
    :param n_epoch: number of epochs to train
    :return:
        model: trained model
    """
    model.train() # prep model for training
    
    
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in tqdm(train_dataloader):

             # Clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            # Calculate the loss
            loss = criterion(output, target)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            optimizer.step()

    return model

```

## D Test the Trained Network
```python
def eval_model(model, dataloader):
    """
    :return:
        Y_pred: prediction of model on the dataloder.
        Y_true: truth labels.
    """
    # Set model to evaluation mode
    model.eval()
    Y_pred = []
    Y_true = []

    # Evaluate the model using on the data in the dataloder
    # Add all the prediction and truth to the corresponding list
    for data, target in dataloader:

        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, dim=1)
        
        Y_pred.append(pred.cpu().numpy())
        Y_true.append(target.cpu().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    return Y_pred, Y_true

```
