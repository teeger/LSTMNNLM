#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "neuralnet_types.h"
#include "neuralnet_map_connection.h"

// <cstdio>
using std::size_t;
// <cmath>
using std::pow;
using std::sqrt;
// <iostream>
using std::cerr;
using std::ostream;
using std::endl;
// <vector>
using std::vector;
// <unordered_map>
using std::unordered_map;
// <algorithm>
using std::fill;
using std::sort;
using std::unique;

namespace neuralnet {


void NeuralNetMapConnection::ForwardPropagate(const NeuralNetLayer &input,
                                           NeuralNetLayerBase &output) {
  assert(input.nneurons() == nneurons_);
  assert(output.nneurons() == nneurons_);
  const vector<ActivationType> &input_ac = input.activations();
  size_t i;
  for (i = 0; i + 7 < nneurons_; ) {
    output.AccumulateInputForActivation(i , input_ac[i]);
    output.AccumulateInputForActivation(i + 1, input_ac[i + 1]);
    output.AccumulateInputForActivation(i + 2, input_ac[i + 2]);
    output.AccumulateInputForActivation(i + 3, input_ac[i + 3]);
    output.AccumulateInputForActivation(i + 4, input_ac[i + 4]);
    output.AccumulateInputForActivation(i + 5, input_ac[i + 5]);
    output.AccumulateInputForActivation(i + 6, input_ac[i + 6]);
    output.AccumulateInputForActivation(i + 7, input_ac[i + 7]);
    i += 8;
  }
  for ( ; i < nneurons_; i++) {
    output.AccumulateInputForActivation(i , input_ac[i]);
  }
}

void NeuralNetMapConnection::BackPropagate(const NeuralNetLayerBase &output,
                                        NeuralNetLayerBase &input) {
  assert(input.nneurons() == nneurons_);
  assert(output.nneurons() == nneurons_);
  const vector<ErrorType> &output_er = output.errors();
  size_t i;
  for (i = 0;i + 7 < nneurons_; ) {
    input.AccumulateInputForError(i, output_er[i]);
    input.AccumulateInputForError(i + 1, output_er[i + 1]);
    input.AccumulateInputForError(i + 2, output_er[i + 2]);
    input.AccumulateInputForError(i + 3, output_er[i + 3]);
    input.AccumulateInputForError(i + 4, output_er[i + 4]);
    input.AccumulateInputForError(i + 5, output_er[i + 5]);
    input.AccumulateInputForError(i + 6, output_er[i + 6]);
    input.AccumulateInputForError(i + 7, output_er[i + 7]);
    i += 8;
  }
  for ( ; i < nneurons_; i++) {
    input.AccumulateInputForError(i, output_er[i]);
  }
}

} // namespace neuralnet
