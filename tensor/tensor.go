package tensor

import (
	"fmt"
	"math"
	"reflect"
)

type Tensor struct {
	number interface{}
	value  reflect.Value
	dtype  reflect.Type
	shape  []int
	name   string
	errors RaiseError{}
}

func NewTensor(a interface{}) *Tensor {
	val := reflect.ValueOf(a)
	var shape []int
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
	}

	return &Tensor{value: val, number: a, shape: shape}
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Name() []int {
	return t.name
}

func (t *Tensor) Dtype() reflect.Type {
	return t.dtype
}

func (t *Tensor) ToValue() *reflect.Value {
	return &reflect.Value(t.number)
}
func Placeholder(shape []int) *Tensor {
	return &Tensor{shape: shape}
}

func TensorFromValue(v reflect.Value) (*Tensor, error){
	if v == nil{
		return nil, fmt.Errorf("You have passed an empty interface (%T)to be transformed. ", v.Addr().Type())
	}
	return &Tensor(value: v, dtype: v.Elem().Type())
}

func AssertionError(x, y *Tensor) error {
	valX, valY := reflect.Value(x.value).Type(), reflect.Value(y.value).Type()
	if valX != valY {
		return fmt.Errorf("Assertion Error: given tensors do not have the same types: %T ----- %T\n", valX, valY)
	}
	for i := range x.shape {
		if x.shape[i] != y.shape[i] {
			return fmt.Errorf("Assertion Error: given tensors do not have the same shapes")
			break
		}

	}
	return nil
}
func (t *Tensor) SetName(name string) {
	t.name = name
}
func printType(a interface{}) {
	fmt.Printf("a: %v\n", a)
}

func (t *Tensor) TensorArray() interface{} {
	typ, arr := t.value, make([]float64, 1)
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		arr = append(arr, val)
	}
	return arr
}

func (*t.Tensor) AtShape(shape []int, a int) interface{} {
	num := t.number
	for _, k := range shape {
		num := num[k]
	}
	return num
}

func (t *Tensor) Add(t2 *Tensor) (*Tensor, error) {
	// CHECK MATCHING SHAPES
	if err := AssertionError(t, t2); err != nil {
		return nil, err
	}
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		count += 1
		t.number += t2.AtShape(shape, count)
	}
	return t, nil
}

func (t *Tensor) Substract(t2 *Tensor) (*Tensor, error) {
	// CHECK MATCHING SHAPES
	if err := AssertionError(t, t2); err != nil {
		return nil, err
	}
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		count += 1
		t.number -= t2.AtShape(shape, count)
	}
	return t, nil
}

func (t *Tensor) Multiply(t2 *Tensor) (*Tensor, error) {
	// CHECK MATCHING SHAPES
	if err := AssertionError(t, t2); err != nil {
		return nil, err
	}
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		count += 1
		t.number *= t2.AtShape(shape, count)
	}
	return t, nil
}

func isZero(x float64) bool {
	return x == 0
}

func isZeroCounts(x float64, count int) int {
	count += x == 0
}
func DivisionByZero() error {
	return fmt.Errorf("Division by Zero present. Inspect your tensors via the Shape, Inspect or Placeholder function. ")
}

func (t *Tensor) Divide(t2 *Tensor) (*Tensor, error) {
	// CHECK MATCHING SHAPES
	if err := AssertionError(t, t2); err != nil {
		return nil, err
	}
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		count += 1
		elem := t2.AtShape(shape, count)
		if isZero(elem) {
			return nil, DivisionByZero()
		}
		t.number /= t2.AtShape(shape, count)
	}
	return t, nil
}

func (t *Tensor) Map(f func(interface{}) interface{}) *Tensor {
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		val = f(val)
	}
	return t.Placeholder(t.value)
}

// a and b are indices for shape
func (t *Tensor) Gather(a, b int) {

}

// follow the tf documentation and implement
func (t *Tensor) ZeroCounts() int {
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		val = typ.Elem()
		count += isZero(val)
	}
	return count
}

func (t* Tensor) GetAxis(axis int) (t*Tensor, error){
	val := t.value

	if t.OutOfAxis(axis) {
		return nil, fmt.Errorf("Axis %d is not present in tensor shape. ", axis, "Validate your tensors with TensorSpec or Shape \n")
	}
	for i:=0;i<axis;i++{
		val = val[i]
	}
	return TensorFromValue(val)
}

func (t *Tensor) OutOfAxis(axis int) bool {
	return axis > len(t.shape)
}

func switch(a,b int) int,int{
	return b,a
}

func (t *Tensor) Gather(a, b, axis int) (*Tensor, error) {
	if t.OutOfAxis(axis) {
		return nil, fmt.Errorf("Axis %d is not present in tensor shape. ", axis, "Validate your tensors with TensorSpec or Shape \n")
	}
	k, err := t.GetAxis(axis)
	if err !=nil{
		return nil, err
	}
	if b>a{
		return nil, fmt.Errorf("In the interval %d to %d you have passed the larger number first. You can also deal with this by calling the switch function\n. ",a,b)
	}
	return k[a:b]

}

func ValueError(value reflect.Value) error{
	ty := value.Type()
	if ty == nil && ty == string{
		return fmt.Errorf("%T does not match Tensor interface", ty)
	}
	tensor := TensorFromValue(value)
	if len(tensor.shape) > 1e6{
		return fmt.Errorf("You have entered an interface size greater than the RAM size: %d", len(tensor.shape))
	}
	
	if _,err:=tensor.GetAxis(0);err!=nil{
		return fmt.Errorf("The given tensor is not valid due to no axis found. Treat carefully.")
	}

	return nil
}

// Squeze removes the dimensison with size 1
func (t* Tensor) Squeeze(){
	typ, count := t.value, 0
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		if val.Len() > 0 {
			val = val.Index(0)
		}
		if shape[0] == 1{
			f := t.number[:][:]
		}
	}
}
func tape(f func(x float64) float64, x float64) float64 {
	epsilon := 1e-6
	deltax := f(x+epsilon) - f(x)
	return deltax / epsilon
}

func sigmoid(x float64) float64 {
	return 1 / (1 - math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

type RaiseError interface {
	Init()
	Check() bool
	Raise() error
}

type ValueError struct {
	tensor *Tensor
	raised bool
}

func (ve *ValueError) Check(tensor Tensor) (bool, error) {
	value := tensor.Value()
	ty := value.Type()
	if ty == nil && ty == string {
		return true, fmt.Errorf("%T does not match Tensor interface", ty)
	}
	if len(tensor.shape) > 1e6 {
		return true, fmt.Errorf("You have entered an interface size greater than the RAM size: %d", len(tensor.shape))
	}

	if _, err := tensor.GetAxis(0); err != nil {
		return true, fmt.Errorf("The given tensor is not valid due to no axis found. Treat carefully.")
	}

	return false, nil
}

func (ve *ValueError) Raise() error {
	raised, err := ve.Check(ve.tensor)
	ve.raised = raised
	if err != nil {
		return fmt.Errorf("ValueError raised. ")
	}
	return nil
}

type AssertionError struct {
	tensor1, tensor2 *Tensor
	raised           bool
}

func (ae *AssertionError) Check(x, y *Tensor) (bool, error) {
	valX, valY := reflect.Value(x.value).Type(), reflect.Value(y.value).Type()
	if valX != valY {
		return true, fmt.Errorf("Assertion Error: given tensors do not have the same types: %T ----- %T\n", valX, valY)
	}
	for i := range x.shape {
		if x.shape[i] != y.shape[i] {
			return true, fmt.Errorf("Assertion Error: given tensors do not have the same shapes")
			break
		}

	}
	return false, nil
}

func (ae *AssertionError) Raise() error {
	raised, err := ae.Check(ae.tensor1, ae.tensor2)
	ae.raised = raised
	if err != nil {
		return fmt.Errorf("Assertion error raised. ")
	}
	return nil
}

type OutOfRangeError struct {
	tensor *Tensor
	a, b   int
}

type UnknownError struct {
	tensor *Tensor
}

func (ue *UnknownError) Check(t *Tensor) (bool,error){
	ue.tensor = t
	if ue.tensor.Value().Type() == nil || ue.tensor.Value().Type() == string{
		return true, fmt.Errorf("%T\n unknown and cannot be converted into Tensor", type(t))
	}
	return false, nil
}

func (ue *UnknownError) Raise() error{
	raised, err := ue.Check(ue.tensor)
	ue.raised = raised
	if err!=nil{
		return fmt.Errorf("Unknown error raised.")			
	}
	return nil
}

// func main() {
// 	// f := [][]int64{
// 	// 	{4, 4},
// 	// 	{1, 1},
// 	// }
// 	l := []int64{1, 3, 4, 5, 4, 3}
// 	t := NewTensor(l)
// 	// k := NewTensor(f)

// 	// fmt.Println(reflect.ValueOf(k.value))
// 	// fmt.Println(AssertionError(t, k))
// 	//var x float64 = 3.4
// 	v := reflect.ValueOf(t.number)
// 	k := v.Type()

// 	fmt.Println(k)
// 	// fmt.Println("kind is float64:", v.Kind() == reflect.Float64)
// 	// fmt.Println("value:", v.Float64())
// 	//n := v.Interface().(v.Type)
// 	value := reflect.New(t.number[0].(reflect.Type)).Interface()
// 	fmt.Println(value)
// 	//fmt.Println(n)
// }



// TODO: gradient tape, poveži z optimizerjem
// uvozi data loaderje pa dodaj malo funkcionalnosti
// tam pri automl