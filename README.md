# NNGo: Neural Network framework in Golang built from scratch

[![Go Report Card](https://goreportcard.com/badge/github.com/timothy102/neuralnetwork)](https://goreportcard.com/report/github.com/timothy102/neuralnetwork)

Neuralnetwork is a hands-on approach to machine learning in Go. For those of you, who have tried Tensorflow for Go or keras, it will fit you perfectly. The framework is still in development, there is still a lot to be implemented. Appreciate every feedback possible. 

## Inspiration Behind NNGo

The primary goal for NNGo is to be a highly performant machine learning/graph computation-based framework. It should bring the appeal of Go (simple compilation and deployment process to the ML world). There is a long way ahead of us regarding deployment, efficiency and managebility, but baby steps, right? :)

The secondary goal for NNGo is to provide a platform for exploration for non-standard deep-learning and neural network related things. Using our framework, you'll be able to expand the horizon of deep learning by exploring the highly abstract tool for extracting the most of the data as well as the algorithms. 

## Installation
```go
go get -u github.com/timothy102/neuralnetwork
import nn "github.com/timothy102/neuralnetwork 

```
## Getting Started

If you seek to know more about the underlying mathematics behind this quite abstract architecture, check the matrix package at: github.com/timothy102/matrix . 
You don't have to worry about it, but as every machine learning engineer knows, it comes in handy. 

```go
model := nn.Sequential([]Layer{
  Conv2D(64,3, 1,Valid),
  MaxPooling2D(2),
  Conv2D(32,3, 1, DefaultPadding),
  MaxPooling2D(2),
  Flatten(),
  Dense(128, Relu),
  Dense(32,Tanh),
  Softmax(10)
}, "sequential")

model.Compile(RMSprop, CrossEntropy, Mae)

history := model.Train(dataX, dataY, numEpochs)
```


## How the Tensor Package Works

Mimicking the Keras architecture, TensorGo works by implementing the unbounded interface method able to reproduce any form or value ensuring tensor scalability. This was accomplished using the `reflect` module in Golang. In order to initialize a tensor, you can either define a placeholder, the tensor constructor or avoid it all together by implementing the higher abstract level of the NNGo library for ML. 

```go
#1
cube := [][][]float64{}
tensor := nn.NewTensor(cube)

#2
shape := []int{2, 3, 4}
t := nn.Placeholder(shape)
```

### Try your first NNGo Program

```go
result := nn.Add(tensor, t)
res := tensor.Add(t)
```

Both solutions yield the same result 😃

## Contact
Please, feel free to reach out on LinkedIn, gmail.
For more, check my medium article. 

`https://towardsdatascience.com/golang-as-the-new-machine-learning-powerforce-e1b74b10b83b`

`https://www.linkedin.com/in/tim-cvetko-32842a1a6/ `

>> cvetko.tim@gmail.com 

## License

Licensed under the MIT [LICENSE](LICENSE)
