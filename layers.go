package neuralnetwork

import (
	"math"
	"math/rand"
	"time"

	"github.com/timothy102/matrix"
)

//Layer interface given these 5 functions which every layer must have.
type Layer interface {
	Call() []float64
	GetWeights() matrix.Matrix
	GetBiases() matrix.Vector
	Name() string
	TrainableParameters() int
}

//DenseLayer defines a fully connected layer.
type DenseLayer struct {
	units             int
	inputs, outputs   []float64
	weights           Weights
	biases            Biases
	trainable         bool
	name              string
	kernelRegularizer func([]float64) []float64
	biasRegularizer   func([]float64) []float64
	Activation        func(float64) float64
	KernelInit        func(float64) float64
	BiasInit          func(float64) float64
}

//Weights struct with the actual kernels and the kernel initializer function.
type Weights struct {
	kernels    matrix.Matrix
	KernelInit func(float64) float64
}

//Biases struct with the actual biases and the bias initializer function.
type Biases struct {
	bs       matrix.Vector
	BiasInit func(float64) float64
}

type shape struct {
	inputShape []float64
}

//WeightInit used for weight initialization. Already defined at the initialization of the dense layer.
func WeightInit(a, b int, kernelInit func(float64) float64) Weights {
	w := matrix.RandomValuedMatrix(a, b).MapFunc(kernelInit)
	return Weights{kernels: w, KernelInit: kernelInit}
}

//BiasInit used for bias initialization. Already defined at the initialization of the dense layer.
func BiasInit(a int, biasInit func(float64) float64) Biases {
	bs := matrix.RandomVector(a).Map(biasInit)
	return Biases{bs: bs, BiasInit: biasInit}
}

//Dense fully connected layer initializer
func Dense(units int, inputs []float64, activation func(float64) float64) DenseLayer {
	weights := WeightInit(units, len(inputs), HeUniform)
	biases := BiasInit(units, ZeroInitializer)
	return DenseLayer{units: units,
		inputs:     inputs,
		Activation: activation,
		weights:    weights,
		biases:     biases,
	}
}

//Call of the dense layer.Outputs the next tensors.
func (d DenseLayer) Call() []float64 {
	vec := matrix.NewVector(d.inputs).ApplyMatrix(d.weights.kernels).Add(d.biases.bs)
	return vec.Map(d.Activation).Slice()
}

//Name of the dense layer
func (d DenseLayer) Name() string {
	return d.name
}

//GetWeights returns the layer's weights.
func (d DenseLayer) GetWeights() matrix.Matrix {
	return d.weights.kernels
}

//GetBiases returns the layer's biases.
func (d DenseLayer) GetBiases() matrix.Vector {
	return d.biases.bs
}

//TrainableParameters returns the count of trainable parameters.
func (d DenseLayer) TrainableParameters() int {
	return d.weights.kernels.NumberOfElements() + d.biases.bs.NumberOfElements()
}

//SetWeights is used for manually defining the weight matrix.
func (d *DenseLayer) SetWeights(kernels matrix.Matrix) {
	d.weights.kernels = kernels
}

//SetBiases is used for manually defining the bias vector.
func (d *DenseLayer) SetBiases(bs matrix.Vector) {
	d.biases.bs = bs
}

//InputLayer layer, much like the keras one.
type InputLayer struct {
	inputs, outputs []float64
	weights         Weights
	biases          Biases
	trainable       bool
	name            string
}

//Input layer
func Input(inputs []float64) InputLayer {
	weights := WeightInit(len(inputs), 1, HeUniform)
	biases := BiasInit(len(inputs), ZeroInitializer)
	return InputLayer{
		inputs:  inputs,
		weights: weights,
		biases:  biases,
	}
}

//Call of the input layer
func (i *InputLayer) Call() []float64 {
	vec := matrix.NewVector(i.inputs).ApplyMatrix(i.weights.kernels).Add(i.biases.bs)
	i.outputs = vec.Slice()
	return vec.Slice()
}

//BatchNormLayer layer
type BatchNormLayer struct {
	inputs, outputs      []float64
	beta, epsilon, alpha float64
	trainable            bool
	name                 string
}

//BatchNorm init
func BatchNorm(inputs []float64) BatchNormLayer {
	return BatchNormLayer{inputs: inputs}
}

//Call for the batch normalization layer
func (bn *BatchNormLayer) Call() []float64 {
	outputs := make([]float64, len(bn.inputs))
	variance := Variance(bn.inputs)
	mean := meanValue(bn.inputs)
	for _, x := range bn.inputs {
		newX := (x - mean) / math.Sqrt(variance+bn.epsilon)
		outputs = append(outputs, bn.alpha*newX+bn.beta)
	}
	bn.outputs = outputs
	return outputs
}

//Variance returns the variance
func Variance(fls []float64) float64 {
	var sum float64
	for _, f := range fls {
		sum += math.Pow(f-meanValue(fls), 2)
	}
	return sum / float64(len(fls))
}

func meanValue(fls []float64) float64 {
	mean := sum(fls) / float64(len(fls))
	return mean
}

//DropoutLayer layer
type DropoutLayer struct {
	inputs []float64
	rate   float64
}

//Dropout init
func Dropout(inputs []float64, rate float64) DropoutLayer {
	return DropoutLayer{inputs: inputs, rate: rate}

}

//Call for the dropout layer
func (dr *DropoutLayer) Call() []float64 {
	weightCount := dr.rate * float64(len(dr.inputs))
	for i := int(weightCount); i > 0; i-- {
		if len(dr.inputs)%int(weightCount) == 0 {
			dr.inputs[i] = 0
		}
	}
	return dr.inputs
}

//SoftmaxLayer layer
type SoftmaxLayer struct {
	inputs, outputs []float64
	classes         int
}

//Softmax returns the softmax layer based on values.
func Softmax(inputs []float64, classes int) SoftmaxLayer {
	return SoftmaxLayer{inputs: inputs, classes: classes}
}

//Call of the softmax
func (s *SoftmaxLayer) Call() []float64 {
	sum := 0.0
	preds := make([]float64, len(s.inputs))
	for i, n := range s.inputs {
		preds[i] -= math.Exp(n - findMax(s.inputs))
		sum += preds[i]
	}
	for k := range preds {
		preds[k] /= sum
	}
	outputs := preds[:s.classes]
	s.outputs = outputs
	return outputs
}

//FlattenLayer layer
type FlattenLayer struct {
	inputs, outputs []float64
	name            string
	trainable       bool
}

//Call of the FlattenLayer
func (f *FlattenLayer) Call() []float64 {
	return f.outputs
}

//Flatten init.
func Flatten(m matrix.Matrix) FlattenLayer {
	return FlattenLayer{outputs: m.ToArray()}
}

//HeUniform stands for He Initialization or the glorot_unifom for kernel_initialization.
func HeUniform(x float64) float64 {
	rand.Seed(time.Now().UnixNano())
	down, upper := x-0.4, x+0.4
	return down + rand.Float64()*(upper-down)
}

//ZeroInitializer returns the zeros initializer for the bias initialization
func ZeroInitializer(x float64) float64 {
	return 0
}

//OnesInitializer returns the ones initializer for the bias initialization
func OnesInitializer(x float64) float64 {
	return 1
}
