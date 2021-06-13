package neuralnetwork

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

type Loader interface {
	Read() []float64
	LoadBatch(batchSize int) []float64
}

type CSVLoader struct {
	file   string
	header bool
	size   int
	data   []float64
}

func (csv *CSVLoader) Read(filepath string) ([]float64, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("cannot open %s", filepath)
	}
	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("cannot read %s", filepath)
	}
	points := make([]float64, len(lines))

	for i, k := range lines {
		points = append(points, strconv.Atoi(strings.Replace(k, ",", "")))
	}
	return points, nil
}

// json loader

type JSONLoader struct {
	file string
	size int
	data []float64
}

func (js *JSONLoader) Read(filepath string) ([]float64, error) {
	js.file = filepath
	jsonFile, err := os.Open(js.file)
	if err != nil {
		return nil, fmt.Errorf("cannot open json file: %s", js.file)
	}
	fmt.Println("Successfully opened %s", js.file)
	defer jsonFile.Close()
	bytes, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		return nil, fmt.Errorf("passed the open stage but cannot read the JSON file %s", filepath)
	}
	var points []float64
	json.Unmarshal([]byte(bytes), &points)

	return points, nil
}
