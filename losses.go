package neuralnetwork

import "math"

//Mse returns the mean squared error between prediction and truth arrays.
func Mse(prediction, truth []float64) float64 {
	loss := 0.0
	for i := range prediction {
		loss += math.Pow(truth[i]-prediction[i], 2)
	}
	return loss
}

//Rmse returns the root mean squared error between prediction and truth arrays.
func Rmse(prediction, truth []float64) float64 {
	return math.Sqrt(Mse(prediction, truth))
}

//SoftDiceLoss returns the soft dice loss based on two sets of values
func SoftDiceLoss(values []float64, truth []float64) float64 {
	var numerator, denominator float64
	for i := range values {
		numerator += 2*(values[i]*truth[i]) - sum(values)
		denominator += (values[i] + truth[i]) - sum(values)
	}
	return 1 - (numerator+1)/(denominator+1)
}

//CrossEntropy returns the cross entropy loss
func CrossEntropy(prediction, truth []float64) float64 {
	var loss float64
	for i := range prediction {
		loss += prediction[i]*math.Log(truth[i]) + (1-prediction[i])*math.Log(1-truth[i])
	}
	return loss
}

func sum(values []float64) float64 {
	var total float64
	for _, v := range values {
		total += v
	}
	return total
}
