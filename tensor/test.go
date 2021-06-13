package matrix

import "testing"

func TestPlaceholder(t *testing.T) {
	shape := int[3,3,2]
	t := Placeholder(shape)
	if t.RaiseError(){
		t.Errorf("Something went wrong with the tensor")
	}
}
