# SGD From Scratch
### We will construct an artificial dataset according to a linear model with additive noise. 
### Our task will be to recover this model’s parameters using the finite set of examples contained in our dataset.

## 1 Generate Synthetic Data

```python
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```
- Creates synthetic data with known ground-truth `w` and `b`, plus noise

---

## 2 Mini-batch Data Iterator

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```
- Shuffles indices, then yields data in mini-batches

---

## 3 Initialize Parameters

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```
- `w` sampled from normal distribution, `b` set to 0
- `requires_grad=True` so gradients can be tracked for both

---

## 4 Linear Model

```python
def linear(X, W, b):
    return torch.matmul(X, W) + b
```
- Implements $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$

---

## 5 Loss Function (Squared Loss)

```python
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()
```

---

## 6 Optimizer (Mini-batch Stochastic Gradient Descent)

```python
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```
- Moves each parameter opposite to its gradient, scaled by the learning rate
- Divides by `batch_size` so step size doesn't depend on batch size
- `torch.no_grad()`: the update itself shouldn't be tracked by autograd
- `.zero_()`: resets gradients after each update (PyTorch accumulates gradients by default)

- Stochastic: estimates the gradient using a randomly sampled subset (a mini-batch) instead of the full dataset
- Gradient Descent: moves parameters in the opposite direction of the gradient to gradually reduce the loss
---

## 7 Training Loop

```python
lr = 0.03
num_epochs = 20
net = linear
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)   # loss on this mini-batch
        l.backward()                # compute gradients
        sgd([w, b], lr, batch_size) # update parameters
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

**Each step:**
- 1 Grab a mini-batch
- 2 Forward pass → compute loss
- 3 Backward pass → compute gradients
- 4 Update parameters with SGD (gradients reset automatically)
- 5 Repeat over all batches and epochs

- After training, `w` and `b` should converge close to `true_w` and `true_b`.
