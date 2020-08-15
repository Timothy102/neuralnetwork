package neuralnetwork

import (
	"math"
)

type activationFunction interface{}

//SoftmaxLayer returns the softmax layer based on values.
func SoftmaxLayer(outputs []float64) []float64 {
	sum := 0.0
	preds := make([]float64, len(outputs))
	for i, n := range outputs {
		preds[i] -= math.Exp(n - findMax(outputs))
		sum += preds[i]
	}
	for k := range preds {
		preds[k] /= sum
	}
	return preds
}
func findMax(fls []float64) float64 {
	max := -math.MaxFloat64
	for _, k := range fls {
		if k > max {
			max = k
		}
	}
	return max
}

//Relu also known as the rectified linear unit activation function.
func Relu(outputs []float64) []float64 {
	preds := make([]float64, len(outputs))
	for _, k := range outputs {
		if k > 0 {
			preds = append(preds, k)
		} else {
			preds = append(preds, 0)
		}
	}
	return preds
}

//Sigmoid function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//SigmoidPrime is the derivative of sigmoid
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

//Tanh returns tanh function.
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

//Elu is exponential linear unit. Alpha is a parameter that should be above 0.
func Elu(x, alpha float64) float64 {
	if x > 0 {
		return x
	}
	return alpha * (math.Exp(x) - 1)
}

//Swish optimizer. Beta should be above 0.
func Swish(x float64, beta float64) float64 {
	return x * Sigmoid(beta*x)
}
