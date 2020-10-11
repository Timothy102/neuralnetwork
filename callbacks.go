package neuralnetwork

import (
	"encoding/csv"
	"fmt"
	"os"
)

//Callback interface
type Callback interface {
	Do(a ...interface{})
}

//EarlyStopper callback
type EarlyStopper struct {
	index    int
	patience int
	minDelta float64
	mode     string
}

//EarlyStopping is much like the keras EarlyStopping callback. Index is the number in the model metrics attribute. Look it up under model.get_metrics()
func (m *Model) EarlyStopping(index int, patience int, at int, minDdelta float64, mode string) {
	metrics := m.GetMetricsByIndex(index)
	values := make([]float64, at)
	for i := 0; i < at; i++ {
		val := metrics.Measure(m.Predict(m.trainDataX), m.trainDataY)
		values = append(values, val)
	}
	for j := range values {
		mind := values[j] - values[j+patience]
		maxd := values[j+patience] - values[j]
		if mode == "auto" && (mind < minDdelta || maxd > minDdelta) {
			m.training = false
		}
		if mode == "min" && mind < minDdelta {
			m.training = false
		}
		if mode == "max" && maxd > minDdelta {
			m.training = false
		}
	}
}

//CsvLogger logger
func (m *Model) CsvLogger(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("could not create file with path %s:%v", filename, err)
	}
	writer := csv.NewWriter(f)
	writer.Write(m.trainingLog)
	return nil
}

//ModelCheckpoint callback
func (m *Model) ModelCheckpoint(filepath string, metrics Metrics, saveWeightsOnly bool) error {
	f, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("could not create file with path %s:%v", filepath, err)
	}
	for _, l := range m.layers {
		s := fmt.Sprintf("%f", l.GetWeights().ToArray())
		f.WriteString(s)
	}
	if !saveWeightsOnly {
		s := fmt.Sprintf("%f", metrics.Measure(m.Predict(m.trainDataX), m.trainDataY))
		f.WriteString(s)
	}
	return nil
}

//CallbackList returns model's callback list
func (m *Model) CallbackList() []Callback {
	return m.callbacks
}

//LearningRateScheduler defined by fn
func (m *Model) LearningRateScheduler(fn func(x float64) float64) {
	m.learningRate = fn(m.learningRate)
}

//CallbackHistory struct
type CallbackHistory struct {
	file *os.File
}

//History for model checkpointing
func (m *Model) History(filepath string) (*CallbackHistory, error) {
	f, err := os.Create(filepath)
	if err != nil {
		return nil, fmt.Errorf("could not create file %s:%v", filepath, err)
	}
	for _, met := range m.modelMetrics {
		val := met.Measure(m.Predict(m.trainDataX), m.trainDataY)
		s := fmt.Sprintf("%f\n", val)
		f.WriteString(s)
	}
	return &CallbackHistory{file: f}, nil
}

//ReadToStrings returns history in strings.
func (ch *CallbackHistory) ReadToStrings() ([]string, error) {
	records, err := csv.NewReader(ch.file).Read()
	if err != nil {
		return nil, fmt.Errorf("Error occured while reading the history file %v", err)
	}
	return records, nil
}

//ReduceLearningRateOnPlateau callback
type ReduceLearningRateOnPlateau struct {
	patience, index int
	factor          float64
	minDelta        float64
	minimumLr       float64
	mode            string
	reducing        bool
}

//ReduceLr callback
func (rlr *ReduceLearningRateOnPlateau) ReduceLr(m *Model, at int) {
	metrics := m.GetMetricsByIndex(rlr.index)
	values := make([]float64, at)
	for i := 0; i < at; i++ {
		val := metrics.Measure(m.Predict(m.trainDataX), m.trainDataY)
		values = append(values, val)
	}
	for j := range values {
		mind := values[j] - values[j+rlr.patience]
		maxd := values[j+rlr.patience] - values[j]
		if rlr.mode == "auto" && (mind < rlr.minDelta || maxd > rlr.minDelta) {
			m.learningRate = m.learningRate * rlr.factor
			rlr.reducing = true
		}
		if rlr.mode == "min" && mind < rlr.minDelta {
			m.learningRate = m.learningRate * rlr.factor
			rlr.reducing = true

		}
		if rlr.mode == "max" && maxd > rlr.minDelta {
			m.learningRate = m.learningRate * rlr.factor
			rlr.reducing = true
		}
		if m.learningRate < rlr.minimumLr {
			m.learningRate = rlr.minimumLr * 100
		}
	}
}
