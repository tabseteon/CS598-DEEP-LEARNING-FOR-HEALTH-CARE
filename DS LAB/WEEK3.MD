#### Previously, we perform mortality prediction based on the last visit's diagnosis codes using DNN.
#### This practice ignores massive information in the previous visits of a patient. Thus, Starting from this lab, we will play with sequential visit data.
#### MLP is quite unsatisfying when dealing with such rich structure data. 
#### This lab introduces convolutional neural networks (CNNs), powerful family of neural networks. Although CNNs are known for image-type data, we will learn how to use CNN for processing the sequential patient data.

## 1 Convolution Operation
#### Convolutional neural networks are efficient architectures for exploring structure in image data.

## 2 Padding and Stride
### Padding
#### Since we typically use small kernels, for any given convolution, we might only lose a few pixels, but this can add up as we apply many successive convolutional layers. One straightforward solution to this problem is to add extra pixels of filler around the boundary of our input image, thus increasing the effective size of the image. Typically, we set the values of the extra pixels to zero. In the figure below, we pad a 3×3 input, increasing its size to 5×5.

### Stride
#### When computing the convolution, we start with the convolution window at the top-left corner of the input tensor, and then slide it over all locations both down and to the right.
#### However, sometimes, either for computational efficiency or because we wish to downsample, we move our window more than one element at a time, skipping the intermediate locations.
#### We refer to the number of rows and columns traversed per slide as the stride. So far, we have used strides of 1, both for height and width. Sometimes, we may want to use a larger stride.

## 3 Multiple Input and Multiple Output Channels

## 4 Pooling
#### Often, as we process images, we want to gradually reduce the spatial resolution of our hidden representations, aggregating information so that the higher up we go in the network, the larger the receptive field (in the input) to which each hidden node is sensitive.
#### Unlike the cross-correlation computation of the inputs and kernels in the convolutional layer, the pooling layer contains no parameters (there is no kernel). Instead, pooling operators are deterministic, typically calculating either the maximum or the average value of the elements in the pooling window.

## Mortality Prediction With CNN
### 1 Padding
#### Note that the first dimension of x can be different for different patients (i.e., different patients will have different number of visits). Thus we need to implement a padding function.
#### The collate function collate_fn() is supposed to pad them into the same shape (2, 3), where 2 is the number of patients, and 3 is the maximum number of visits.
#### We need to pad the sequences into the same length so that we can do batch training on GPU, which will run much faster. Or, if they have different length, we have to process them one by one. This is extremely slow, especially with a large dataset.
#### You may also wonder will this padding add some extra noise to the dataset (because we change the number of visits for some patients). The answer is: it depends. Sometimes, padding will bring in some noise and we need to have a separate mask to remove the noise later (you will see this in the next lab).

### 2 Build the Model
#### Now, let us build a 1D CNN model. For each patient, the CNN model will take an input tensor of shape (# of visits, total # of codes), and produce an output tensor of 1-dim (0 for non-mortality, 1 for moratality)

### 3 Train the Network

```python

for epoch in range(n_epochs):
    
    train_loss = 0
    for x, y in train_loader:
        """ Step 1. clear gradients """
        optimizer.zero_grad()
        """  Step 2. perform forward pass using `model`, save the output to y_hat """
        y_hat = model(x)
        """ Step 3. calculate the loss using `criterion`, save the output to loss. """
        # convert shape from [batch size, 1] to [batch size]
        y_hat = y_hat.view(y_hat.shape[0])
        loss = criterion(y_hat, y)
        """ Step 4. backward pass """
        loss.backward()
        """ Step 5. optimization """
        optimizer.step()
        """ Step 6. record loss """
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    evaluate(model, train_loader)
    evaluate(model, test_loader)

```

# CNN 모델 학습 파이프라인

| 단계 | 항목 | 설명 |
|------|------|------|
| 1 | **모델 빌드** | `Net` 클래스 정의 및 인스턴스화 (`model = Net()`) |
| 2 | **Loss function 정의** | `criterion` 정의 (예: `nn.BCELoss()`) |
| 3 | **Optimizer 정의** | `optimizer` 정의 (예: `torch.optim.SGD(model.parameters(), lr=0.01)`) |
| 4 | **Training loop** | epoch 반복 |
| 4-1 | Batch 반복 | 각 epoch 안에서 batch 단위 반복 |
| 4-1-1 | Forward pass | 입력을 모델에 통과시켜 예측값 계산 |
| 4-1-2 | Loss 계산 | 예측값과 정답을 비교하여 손실 계산 |
| 4-1-3 | Backward pass | gradient 계산 (`loss.backward()`) |
| 4-1-4 | Optimizer step | 파라미터 업데이트 (`optimizer.step()`) |
| 5 | **Evaluation loop** (선택) | 학습 후 성능 확인 |
