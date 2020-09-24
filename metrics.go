package neuralnetwork

import "math"

//JaccardIndex returns the Jaccard index.
func JaccardIndex(predicted, actual []float64) int {
	var sum int
	for i := range predicted {
		if predicted[i] == actual[i] {
			sum++
		}
	}
	return sum / len(predicted)
}

//F1Score
func F1Score(predicted, actual []float64) int {
	return 2 * (Precision(predicted, actual) * Recall(predicted, actual)) / (Precision(predicted, actual) + Recall(predicted, actual))
}

//Sensitivity returns the sensitivity
func Sensitivity(predicted, actual []float64) int {
	tp := TruePositivies(predicted, actual)
	fn := FalseNegatives(predicted, actual)
	return tp / (tp + fn)
}

//Specificity returns the specificity
func Specificity(predicted, actual []float64) int {
	fp := FalsePositives(predicted, actual)
	tn := TrueNegatives(predicted, actual)
	return fp / (fp + tn)
}

//Precision returns the precision.
func Precision(predicted, actual []float64) int {
	tp := TruePositivies(predicted, actual)
	fp := FalsePositives(predicted, actual)
	return tp / (tp + fp)
}

//Recall returns the recall.
func Recall(predicted, actual []float64) int {
	tp := TruePositivies(predicted, actual)
	fn := FalseNegatives(predicted, actual)
	return tp / (tp + fn)
}

//TruePositivies returns the number of true positive predicted values.
func TruePositivies(predicted, actual []float64) int {
	var sum int
	for i := range predicted {
		if predicted[i] == actual[i] {
			sum++
		}
	}
	return sum
}

//TrueNegatives returns the number of true negative predicted values.
func TrueNegatives(predicted, actual []float64) int {
	var sum int
	for i := range predicted {
		if predicted[i] == actual[i] {
			sum++
		}
	}
	return sum
}

//FalsePositives returns the number of false positive predicted values.
func FalsePositives(predicted, actual []float64) int {
	var sum int
	for i := range predicted {
		if predicted[i] == actual[i] {
			sum++
		}
	}
	return sum
}

//FalseNegatives returns the number of false negative predicted values.
func FalseNegatives(predicted, actual []float64) int {
	var sum int
	for i := range predicted {
		if predicted[i] == actual[i] {
			sum++
		}
	}
	return sum
}

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

//SoftDiceLoss implements the soft dice loss.
func SoftDiceLoss(values []float64, truth []float64) float64 {
	var numerator, denominator float64
	for i := range values {
		numerator += 2*(values[i]*truth[i]) - sum(values)
		denominator += (values[i] + truth[i]) - sum(values)
	}
	return 1 - (numerator+1)/(denominator+1)
}
func sum(values []float64) float64 {
	var total float64
	for _, v := range values {
		total += v
	}
	return total
}

//RidgeRegression returns the RidgeRegression or the l2 regularization to the loss function.
func RidgeRegression(actual, pred []float64, lambda float64) float64 {
	var loss float64
	var l2 float64
	for i := range actual {

		loss += math.Pow(pred[i]-actual[i], 2)
		l2 += lambda * math.Pow(actual[i], 2)
	}
	return loss + l2
}

//LassoRegression returns the LassoRegression or the l1 regularization to the loss function.
func LassoRegression(actual, pred []float64, lambda float64) float64 {
	var loss float64
	var l1 float64
	for i := range actual {

		loss += math.Pow(pred[i]-actual[i], 2)
		l1 += lambda * math.Abs(actual[i])
	}
	return loss + l1
}

//CrossEntropy returns the cross entropy loss
func CrossEntropy(prediction, truth []float64) float64 {
	var loss float64
	for i := range prediction {
		loss += prediction[i]*math.Log(truth[i]) + (1-prediction[i])*math.Log(1-truth[i])
	}
	return loss
}
