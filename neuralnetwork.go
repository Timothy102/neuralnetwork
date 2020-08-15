//Package neuralnetwork implements the simple neural network  architecture.
package neuralnetwork

import "github.com/timothy102/matrix"

//Network defines the neural network.
type Network struct {
	inputNodes, hiddenNodes, outputNodes int
	weightsIh, weightsHo, biasO, biasH   matrix.Matrix
	learningRate                         float64
}

//InitNetwork initializes the network with the number of nodes and the learning rate.
func InitNetwork(inputNodes, hiddenNodes, outputNodes int, lr float64) Network {
	weightsIh := matrix.RandomValuedMatrix(hiddenNodes, inputNodes)
	weightsHo := matrix.RandomValuedMatrix(outputNodes, hiddenNodes)
	biasO := matrix.RandomValuedMatrix(outputNodes, 1)
	biasH := matrix.RandomValuedMatrix(hiddenNodes, 1)

	return Network{inputNodes: inputNodes,
		hiddenNodes:  hiddenNodes,
		outputNodes:  outputNodes,
		weightsIh:    weightsIh,
		weightsHo:    weightsHo,
		biasH:        biasH,
		biasO:        biasO,
		learningRate: lr,
	}
}

//Train performs the training.
func (n *Network) Train(inputArray, targetArray []float64) {
	inputs := matrix.FromArray(inputArray)
	hidden := n.weightsIh.Multiply(inputs)
	hidden.Add(n.biasH)
	hidden.MapFunc(matrix.Sigmoid)

	output := n.weightsHo.Multiply(hidden)
	output.Add(n.biasO)
	output.MapFunc(matrix.Sigmoid)

	//Turn targets into matrix.
	targets := matrix.FromArray(targetArray)

	//Calculate error->still a matrix of values.
	absErrors := targets.Subtract(output)

	//Calculate gradient
	gradients := output.MapFunc(matrix.SigmoidPrime)
	gradients.Multiply(absErrors)
	gradients.ScalarAdition(n.learningRate)

	//Derivatives
	devHidden := hidden.Transpose()
	weightsHoDerivative := gradients.Multiply(devHidden)

	// Adjust the weights by deltas
	n.weightsHo.Add(weightsHoDerivative)
	n.biasO.Add(gradients)

	// Calculate the hidden layer errors
	hiddenlayerError := n.weightsHo.Transpose()
	hiddenErrors := hiddenlayerError.Multiply(absErrors)
	hiddenG := hidden.MapFunc(matrix.Sigmoid)
	hiddenG.Multiply(hiddenErrors)
	hiddenG.ScalarMultiplication(n.learningRate)

	inputsTranspose := inputs.Transpose()
	weightIHDeltas := hiddenG.Multiply(inputsTranspose)
	n.weightsIh.Add(weightIHDeltas)
	n.biasH.Add(hiddenG)

	output.PrintByRow()
	targets.PrintByRow()
}

//Predict returns the model's prediction based on inputArray
func (n *Network) Predict(inputArray []float64) []float64 {
	inputs := matrix.FromArray(inputArray)
	hidden := n.weightsIh.Multiply(inputs)
	hidden.Add(n.biasH)
	hidden.MapFunc(matrix.Sigmoid)

	output := n.weightsHo.Multiply(hidden)
	output.Add(n.biasO)
	output.MapFunc(matrix.Sigmoid)
	return output.ToArray()
}
