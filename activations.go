package neuralnetwork

import (
	"math"
)

//Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//SigmoidPrime is the derivative of the Sigmoid
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

//Tanh returns the tanh activation function.
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

//Relu implements the rectified linear unit.
func Relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func findMax(fls []float64) float64 {
	max := -10000.0
	for _, k := range fls {
		if k > max {
			max = k
		}
	}
	return max
}

//Elu is an activation function. 0.7 is a parameter that should be above 0
func Elu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.7 * (math.Exp(x) - 1)
}

//Swish activation function. Beta is a parameter that should be above 0
func Swish(x float64) float64 {
	beta := 0.8
	return x * Sigmoid(beta*x)
}
