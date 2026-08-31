#### Previously, we perform mortality prediction based on the last visit's diagnosis codes using DNN.
#### This practice ignores massive information in the previous visits of a patient. Thus, Starting from this lab, we will play with sequential visit data.
#### MLP is quite unsatisfying when dealing with such rich structure data. 
#### This lab introduces convolutional neural networks (CNNs), powerful family of neural networks. Although CNNs are known for image-type data, we will learn how to use CNN for processing the sequential patient data.

## 1 Convolution Operation
#### Convolutional neural networks are efficient architectures for exploring structure in image data.
