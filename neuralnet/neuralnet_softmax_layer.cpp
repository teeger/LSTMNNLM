#include <cstdlib>
#include <vector>
#include "neuralnet_softmax_layer.h"

using std::size_t;
using std::vector;

namespace neuralnet {

void NeuralNetSoftmaxLayer::ComputeActivationsImpl() {
  double sum = 0.0;
  size_t i;
  size_t n = nneurons();
  const vector<ActivationInputType> &acin = activationinputs();
  for (i = 0; i + 7 < n;) {
    aux_activations_[i] = stable_fast_exp(acin[i]);
    aux_activations_[i + 1] = stable_fast_exp(acin[i + 1]);
    aux_activations_[i + 2] = stable_fast_exp(acin[i + 2]);
    aux_activations_[i + 3] = stable_fast_exp(acin[i + 3]);
    aux_activations_[i + 4] = stable_fast_exp(acin[i + 4]);
    aux_activations_[i + 5] = stable_fast_exp(acin[i + 5]);
    aux_activations_[i + 6] = stable_fast_exp(acin[i + 6]);
    aux_activations_[i + 7] = stable_fast_exp(acin[i + 7]);

    sum += aux_activations_[i];
    sum += aux_activations_[i + 1];
    sum += aux_activations_[i + 2];
    sum += aux_activations_[i + 3];
    sum += aux_activations_[i + 4];
    sum += aux_activations_[i + 5];
    sum += aux_activations_[i + 6];
    sum += aux_activations_[i + 7];

    i += 8;
  }

  for (; i < n; ++i) {
    aux_activations_[i] = stable_fast_exp(acin[i]);
    sum += aux_activations_[i];
  }

  // normalize
  for (i = 0; i + 7 < n;) {
    set_activations(i, aux_activations_[i] / sum);
    set_activations(i + 1, aux_activations_[i + 1] / sum);
    set_activations(i + 2, aux_activations_[i + 2] / sum);
    set_activations(i + 3, aux_activations_[i + 3] / sum);
    set_activations(i + 4, aux_activations_[i + 4] / sum);
    set_activations(i + 5, aux_activations_[i + 5] / sum);
    set_activations(i + 6, aux_activations_[i + 6] / sum);
    set_activations(i + 7, aux_activations_[i + 7] / sum);

    i += 8;
  }

  for (; i < n; ++i) {
    set_activations(i, aux_activations_[i] / sum);
  }
}

} // namespace neuralnet
