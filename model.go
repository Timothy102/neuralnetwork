package neuralnetwork

import (
	"fmt"
	"time"
)

//Model implements the model architecure.
type Model struct {
	layers            []Layer
	name              string
	optimizer         Optimizer
	loss              func([]float64, []float64) float64
	lossValues        []float64
	training_duration time.Duration
	modelMetrics      []Metrics
}

//Metrics is an interface that requires two functions, Measure and Name and is passed to the model.compile method.
type Metrics interface {
	Measure([]float64, []float64) float64
	Name() string
}

//Optimizer interface requires an ApplyGradients function. Pass it to the model compilation.
type Optimizer interface {
	ApplyGradients()
}

//Sequential returns a model given layers and a name.
func Sequential(layers []Layer, name string) *Model {
	return &Model{layers: layers, name: name}
}

//Add method adds a layer to the end of the model architecture
func (m *Model) Add(layer Layer) *Model {
	m.layers[len(m.layers)] = layer
	return m
}

//GetLayerByIndex returns the ith layer.
func (m *Model) GetLayerByIndex(index int) Layer {
	return m.layers[index]
}

//GetLayerByName returns the layer given its name.
func (m *Model) GetLayerByName(name string) Layer {
	for i := range m.layers {
		if m.layers[i].Name() == name {
			return m.layers[i]
		}
	}
	return m.layers[0]
}

//Compile compiles the model given the optimizer, loss and metrics
func (m *Model) Compile(optimizer Optimizer, loss func([]float64, []float64) float64, ms []Metrics) {
	m.optimizer = optimizer
	m.loss = loss
	m.modelMetrics = ms
}

//Predict does the feed forward magic when fed the inputs.
func (m *Model) Predict(values []float64) []float64 {
	var outputs []float64
	for i := range m.layers {
		outputs = m.layers[i].Call()
		m.layers[i+1].Call()
		if i == len(m.layers)-1 {
			return outputs
		}
	}
	return outputs
}

//Train trains the model given trainX and  trainY data and the number of epochs. It keeps track of the defined metrics and prints it every epoch. It also prints the training duration.
//It returns a map from strings to floats, where strings represent the metrics name and float the metrics value.
func (m *Model) Train(trainX, trainY []float64, epochs int) map[string]float64 {
	startTime := time.Now()
	metricsValues := make(map[string]float64, len(m.modelMetrics))
	for i := 1; i < epochs; i++ {
		for j := 0; j < len(trainX); j++ {
			lossValue := m.loss(m.Predict(trainX), trainY)
			m.lossValues = append(m.lossValues, lossValue)
			m.optimizer.ApplyGradients()
		}
		avg := MeanValue(m.lossValues)
		for _, met := range m.modelMetrics {
			metricsValues[met.Name()] = met.Measure(m.Predict(trainX), trainY)
		}
		fmt.Printf("Epoch: %d		Loss:%.4f\n", i, avg)
	}
	endTime := time.Now()
	m.training_duration = endTime.Sub(startTime)
	fmt.Printf("Training duration: %s\n", m.training_duration.String())
	return metricsValues
}

//Summary prints the layer by layer summaary along with trainable parameters.
func (m *Model) Summary() {
	var sum int
	for i := range m.layers {
		tp := m.layers[i].TrainableParameters()
		sum += tp
		fmt.Printf("name: %s		trainable parameters: %d\n", m.layers[i].Name(), tp)
	}
	fmt.Println("Trainable parameters: ", sum)
}
