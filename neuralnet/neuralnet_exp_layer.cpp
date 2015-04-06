#include <cstdlib>
#include <vector>
#include "neuralnet_numeric.h"
#include "neuralnet_exp_layer.h"

using std::size_t;
using std::vector;

namespace neuralnet {

void NeuralNetExpLayer::ComputeActivationsImpl() {
  size_t i;
  const size_t n = nneurons();
  const vector<ActivationInputType> &acin = activationinputs();
  for (i = 0; i + 7 < n;) {
    set_activations(i, stable_fast_exp(acin[i]));
    set_activations(i + 1, stable_fast_exp(acin[i + 1]));
    set_activations(i + 2, stable_fast_exp(acin[i + 2]));
    set_activations(i + 3, stable_fast_exp(acin[i + 3]));
    set_activations(i + 4, stable_fast_exp(acin[i + 4]));
    set_activations(i + 5, stable_fast_exp(acin[i + 5]));
    set_activations(i + 6, stable_fast_exp(acin[i + 6]));
    set_activations(i + 7, stable_fast_exp(acin[i + 7]));
    i += 8;
  }
  for (; i < n; ++i) {
    set_activations(i, stable_fast_exp(acin[i]));
  }
}

} // namespace neuralnet
