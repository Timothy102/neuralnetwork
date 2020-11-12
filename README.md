#### neuralnetwork

Neuralnetwork is a hands-on approach to machine learning in Go. For those of you, who have tried Tensorflow for Go or keras, it will fit you perfectly. The framework is still in development, there is still a lot to be implemented. Appreciate every feedback possible. 

## Installation
```go
go get -u github.com/timothy102/neuralnetwork
import net "github.com/timothy102/neuralnetwork 

```
## Usage

If you seek to know more about the underlying mathematics behind this quite abstract architecture, check the matrix package at: github.com/timothy102/matrix . 
You don't have to worry about it, but as every machine learning engineer knows, it comes in handy. 

```go
model := Sequential([]Layer{
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

## Contact
Please, feel free to reach out on LinkedIn, gmail.
For more, check my medium article. 
https://towardsdatascience.com/golang-as-the-new-machine-learning-powerforce-e1b74b10b83b
https://www.linkedin.com/in/tim-cvetko-32842a1a6/

cvetko.tim@gmail.com
