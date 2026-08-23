## DNN in PyTorch: We will perform mortality prediction based on the last visit's diagnosis codes using DNN.

### Nonlinearity
### DNN with Pytorch
### Preprocessing
### DNN Model

## 1 Nonlinearity
### 1.1 Linear Models May Go Wrong
#### Say for example that we want to predict probability of death based on body temperature. 
#### For individuals with a body temperature above 37°C, higher temperatures indicate greater risk. 
#### However, for individuals with body temperatures below 37°C, higher temperatures indicate lower risk.

### 1.2 Incorporating Hidden Layers
#### This architecture is commonly called a multilayer perceptron, often abbreviated as MLP.

### 1.3 From Linear to Nonlinear
#### ReLU Function
#### The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the rectified linear unit (ReLU).

### 2 DNN Model
#### 

### Mortality Prediction
### 1 Preprocessing
#### The goal is to extract the diagnosis codes for each admission (model input) and the mortality information (model output).
### 2 DNN model
#### In the previous lab, we implement the linear regression model, which only has one layer. 
#### Thanks to the increasing amount of data and growing computing power, deep learning networks tend to be massive with dozens or hundreds of layers, that is where the term "deep" comes from.
#### PyTorch has a nice module nn that provides a nice way to efficiently build large neural networks.

```python
# number of epochs to train the model
# feel free to change this
n_epochs = 60

# prep model for training
model.train()

for epoch in range(n_epochs):

    train_loss = 0
    for x, y in train_loader:
        # Step 1. Clear gradients from the previous batch
        optimizer.zero_grad()

        # Step 2. Forward pass — compute model predictions
        y_hat = model(x)
        y_hat = y_hat.view(y_hat.shape[0])  # [batch_size, 1] -> [batch_size]

        # Step 3. Compute the loss between predictions and true labels
        loss = criterion(y_hat, y.float())

        # Step 4. Backward pass — compute gradients w.r.t. model parameters
        loss.backward()

        # Step 5. Optimization step — update model parameters
        optimizer.step()

        # Step 6. Accumulate the batch loss
        train_loss += loss.item()

    # Average training loss over all batches in this epoch
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

    # Evaluate model performance on the test set after each epoch
    evaluate(model, test_loader)
```

## Step-by-Step Explanation

| Step | Code | What it does |
|------|------|---------------|
| 1 | `optimizer.zero_grad()` | Clears gradients accumulated from the previous batch. PyTorch accumulates gradients by default, so this must be reset every iteration. |
| 2 | `y_hat = model(x)` | Runs the forward pass — feeds the input batch `x` through the network to get predictions. |
| 3 | `loss = criterion(y_hat, y.float())` | Computes how far off the predictions are from the true labels using the loss function (e.g. `BCELoss`). |
| 4 | `loss.backward()` | Performs backpropagation — computes the gradient of the loss with respect to every model parameter. |
| 5 | `optimizer.step()` | Updates model parameters using the computed gradients (e.g. via SGD). |
| 6 | `train_loss += loss.item()` | Accumulates the batch loss for logging/monitoring. |

