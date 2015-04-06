#include <cstdlib>
#include <vector>
#include "neuralnet_centered_sigmoid_layer.h"

using std::size_t;
using std::vector;

namespace neuralnet {

void NeuralNetCenteredSigmoidLayer::ComputeActivationsImpl() {
  size_t i;
  const size_t n = nneurons();
  const vector<ActivationInputType> &acin = activationinputs();
  for (i = 0; i + 7 < n;) {
    set_activations(i, centered_sigmoid_ac_func(acin[i]));
    set_activations(i + 1, centered_sigmoid_ac_func(acin[i + 1]));
    set_activations(i + 2, centered_sigmoid_ac_func(acin[i + 2]));
    set_activations(i + 3, centered_sigmoid_ac_func(acin[i + 3]));
    set_activations(i + 4, centered_sigmoid_ac_func(acin[i + 4]));
    set_activations(i + 5, centered_sigmoid_ac_func(acin[i + 5]));
    set_activations(i + 6, centered_sigmoid_ac_func(acin[i + 6]));
    set_activations(i + 7, centered_sigmoid_ac_func(acin[i + 7]));
    i += 8;
  }
  for (; i < n; ++i) {
    set_activations(i, centered_sigmoid_ac_func(acin[i]));
  }
}

void NeuralNetCenteredSigmoidLayer::ComputeErrorsImpl() {
  size_t i;
  size_t n = nneurons();
  const vector<ErrorInputType> &erin = errorinputs();
  const vector<ActivationType> &ac = activations();
  for (i = 0; i + 7 < n;) {
    set_errors(i, centered_sigmoid_er_func(erin[i], ac[i]));
    set_errors(i + 1, centered_sigmoid_er_func(erin[i + 1], ac[i + 1]));
    set_errors(i + 2, centered_sigmoid_er_func(erin[i + 2], ac[i + 2]));
    set_errors(i + 3, centered_sigmoid_er_func(erin[i + 3], ac[i + 3]));
    set_errors(i + 4, centered_sigmoid_er_func(erin[i + 4], ac[i + 4]));
    set_errors(i + 5, centered_sigmoid_er_func(erin[i + 5], ac[i + 5]));
    set_errors(i + 6, centered_sigmoid_er_func(erin[i + 6], ac[i + 6]));
    set_errors(i + 7, centered_sigmoid_er_func(erin[i + 7], ac[i + 7]));
    i += 8;
  }
  for (; i < n; ++i) {
    set_errors(i, centered_sigmoid_er_func(erin[i], ac[i]));
  }
}

} // namespace neuralnet
