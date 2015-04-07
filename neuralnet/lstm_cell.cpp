#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include <limits>
#include "futil.h"
#include "lstm_cell.h"

// <cstdio>
using std::size_t;
// <cstdlib>
using std::exit;
// <cmath>
using std::log;
using std::exp;
using std::sqrt;
using std::abs;
// <ctime>
using std::clock_t;
using std::clock;
// <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
// <fstream>
using std::ifstream;
using std::ofstream;
// <ios>
using std::ios;
// <string>
using std::string;
using std::to_string;
// <vector>
using std::vector;
// <limits>
using std::numeric_limits;

namespace neuralnet {

void NeuralNetLSTMCell::AllocateModel() {
  // lstm should have the same dimension as the hidden layer
  memory_cell_layers_.set_capacity(bptt_unfold_level_ + 1);
  for(size_t i = 0; i < bptt_unfold_level_ + 1; i++) {
    memory_cell_layers_[i].set_nneurons(num_cells_, false, false);
    memory_cell_layers_[i].set_errorinput_cutoff(errorinput_cutoff_);
  }
  //input_gate_layer_.set_nneurons(num_cells_, false, false);
  //output_gate_layer_.set_nneurons(num_cells_, false, false);
  //forget_gate_layer_.set_nneurons(num_cells_, false, false);
  //input_layer_.set_nneurons(num_cells_, false, false);
  mc_output_layers_.set_nneurons(num_cells_, false, false);

  connection_cell_ig_.set_dims(true, num_cells_, num_cells_);
  last_connection_cell_ig_.set_dims(true, num_cells_, num_cells_);
  connection_cell_ig_.set_adagrad(adagrad_);
  last_connection_cell_ig_.set_adagrad(adagrad_);
  connection_cell_ig_.set_l2_regularization_param(l2_regularization_param_);
  last_connection_cell_ig_.set_l2_regularization_param(l2_regularization_param_);

  connection_cell_og_.set_dims(true, num_cells_, num_cells_);
  last_connection_cell_og_.set_dims(true, num_cells_, num_cells_);
  connection_cell_og_.set_adagrad(adagrad_);
  last_connection_cell_og_.set_adagrad(adagrad_);
  connection_cell_og_.set_l2_regularization_param(l2_regularization_param_);
  last_connection_cell_og_.set_l2_regularization_param(l2_regularization_param_);

  connection_cell_fg_.set_dims(true, num_cells_, num_cells_);
  last_connection_cell_fg_.set_dims(true, num_cells_, num_cells_);
  connection_cell_fg_.set_adagrad(adagrad_);
  last_connection_cell_fg_.set_adagrad(adagrad_);
  connection_cell_fg_.set_l2_regularization_param(l2_regularization_param_);
  last_connection_cell_fg_.set_l2_regularization_param(l2_regularization_param_);

  map_connection_.set_nneurons(num_cells_);

}

void NeuralNetLSTMCell::ResetActivations() {
  // Initialize memory cell layer activations with zeros.
  for(size_t i = 0; i < bptt_unfold_level_ + 1; i++) {
    memory_cell_layers_[i].SetActivationsToValue(0.0f);
  }

  // Initialize all the gates and input layer with zeros.
  mc_output_layers_.SetActivationsToValue(0.0f);

}

void NeuralNetLSTMCell::InitializeCell() {
  // Reset activations.
  ResetActivations();

  // Randomly initialize cell to gates connections.
  connection_cell_ig_.RandomlyInitialize(rng_engine_);
  connection_cell_og_.RandomlyInitialize(rng_engine_);
  connection_cell_fg_.RandomlyInitialize(rng_engine_);

  // The connection_bias_output_ and connection_globalbias_output_ are initialized as 0s.

  // The connections for max-ent features are initialized as 0s.
  // Note: at least connection_input_output_ is initialized as all 0s.
}

void NeuralNetLSTMCell::GateForwardPropagate(const NeuralNetLayer &input, const NeuralNetLayer &gate_input, NeuralNetLayerBase &output) {
  assert(input.nneurons() == gate_input.nneurons());
  assert(input.nneurons() == output.nneurons());
  assert(input.nneurons() == num_cells_);

  const vector<ActivationType> &input_ac = input.activations();
  const vector<ActivationType> &gate_input_ac = gate_input.activations();

  size_t i;
  for (i = 0; i + 7 < num_cells_; ) {
    output.AccumulateInputForActivation(i, input_ac[i] * gate_input_ac[i]);
    output.AccumulateInputForActivation(i + 1, input_ac[i + 1] * gate_input_ac[i + 1]);
    output.AccumulateInputForActivation(i + 2, input_ac[i + 2] * gate_input_ac[i + 2]);
    output.AccumulateInputForActivation(i + 3, input_ac[i + 3] * gate_input_ac[i + 3]);
    output.AccumulateInputForActivation(i + 4, input_ac[i + 4] * gate_input_ac[i + 4]);
    output.AccumulateInputForActivation(i + 5, input_ac[i + 5] * gate_input_ac[i + 5]);
    output.AccumulateInputForActivation(i + 6, input_ac[i + 6] * gate_input_ac[i + 6]);
    output.AccumulateInputForActivation(i + 7, input_ac[i + 7] * gate_input_ac[i + 7]);
    i += 8;
  }
  for ( ; i < num_cells_; i++) {
    output.AccumulateInputForActivation(i, input_ac[i] * gate_input_ac[i]);
  }
}

void NeuralNetLSTMCell::ForwardPropagate(NeuralNetLayer &input_layer, NeuralNetLayer &input_gate_layer, NeuralNetLayer &output_gate_layer, NeuralNetLayer &forget_gate_layer, NeuralNetLayerBase &hidden_layer) {
  assert(input_layer.nneurons() == num_cells_);
  assert(input_gate_layer.nneurons() == num_cells_);
  assert(output_gate_layer.nneurons() == num_cells_);
  assert(forget_gate_layer.nneurons() == num_cells_);
  assert(hidden_layer.nneurons() == num_cells_);
  
  input_layer.ComputeActivations();

  // last memory cell -> input gate
  // last memory cell -> forget gate
  connection_cell_ig_.ForwardPropagate(memory_cell_layers_[0], input_gate_layer);
  input_gate_layer.ComputeActivations();
  connection_cell_fg_.ForwardPropagate(memory_cell_layers_[0], forget_gate_layer);
  forget_gate_layer.ComputeActivations();
  
  // propagate input -> memory cell and last memory cell -> memory cell
  memory_cell_layers_.rotate(bptt_unfold_level_);
  neuralnet::NeuralNetIdentityLayer &current_memory_cell_layer = memory_cell_layers_[0];
  current_memory_cell_layer.ResetInputForActivations();
  GateForwardPropagate(input_layer, input_gate_layer, current_memory_cell_layer);
  // propagate forget gate -> memory cell
  GateForwardPropagate(memory_cell_layers_[1], forget_gate_layer, current_memory_cell_layer);
  current_memory_cell_layer.ComputeActivations();

  // propagate current memory cell -> output gate cell
  // Note:: reset activations should be carried out outside the lstm cell
  connection_cell_og_.ForwardPropagate(current_memory_cell_layer, output_gate_layer);
  output_gate_layer.ComputeActivations();

  // current memory cell -> mc output layer
  mc_output_layers_.ResetInputForActivations();
  map_connection_.ForwardPropagate(current_memory_cell_layer, mc_output_layers_);
  mc_output_layers_.ComputeActivations();

  // current memory cell -> hidden layer
  GateForwardPropagate(mc_output_layers_, output_gate_layer, hidden_layer);
}


void NeuralNetLSTMCell::FastUpdateWeightsMajor(float learning_rate) {
  connection_cell_ig_.FastUpdateWeightsMajor(learning_rate);
  connection_cell_og_.FastUpdateWeightsMajor(learning_rate);
  connection_cell_fg_.FastUpdateWeightsMajor(learning_rate);
}

void NeuralNetLSTMCell::FastUpdateWeightsMinor() {
  connection_cell_ig_.FastUpdateWeightsMinor();
  connection_cell_og_.FastUpdateWeightsMinor();
  connection_cell_fg_.FastUpdateWeightsMinor();
}

void NeuralNetLSTMCell::GateBackPropagate(const NeuralNetLayerBase &output, NeuralNetLayer &input, NeuralNetLayer &input_gate) {
  assert(input.nneurons() == input_gate.nneurons());
  assert(input.nneurons() == output.nneurons());
  assert(input.nneurons() == num_cells_);

  const vector<ErrorType> &output_er = output.errors();
  const vector<ActivationType> & input_ac = input.activations();
  const vector<ActivationType> & ig_ac = input_gate.activations();
  size_t i;
  for (i = 0; i + 7 < num_cells_; ) {
    // backpropgate to input
    input.AccumulateInputForError(i, output_er[i] * ig_ac[i]);
    input.AccumulateInputForError(i + 1, output_er[i + 1] * ig_ac[i + 1]);
    input.AccumulateInputForError(i + 2, output_er[i + 2] * ig_ac[i + 2]);
    input.AccumulateInputForError(i + 3, output_er[i + 3] * ig_ac[i + 3]);
    input.AccumulateInputForError(i + 4, output_er[i + 4] * ig_ac[i + 4]);
    input.AccumulateInputForError(i + 5, output_er[i + 5] * ig_ac[i + 5]);
    input.AccumulateInputForError(i + 6, output_er[i + 6] * ig_ac[i + 6]);
    input.AccumulateInputForError(i + 7, output_er[i + 7] * ig_ac[i + 7]);
    // backpropgate to input gate
    input_gate.AccumulateInputForError(i, output_er[i] * input_ac[i]);
    input_gate.AccumulateInputForError(i + 1, output_er[i + 1] * input_ac[i + 1]);
    input_gate.AccumulateInputForError(i + 2, output_er[i + 2] * input_ac[i + 2]);
    input_gate.AccumulateInputForError(i + 3, output_er[i + 3] * input_ac[i + 3]);
    input_gate.AccumulateInputForError(i + 4, output_er[i + 4] * input_ac[i + 4]);
    input_gate.AccumulateInputForError(i + 5, output_er[i + 5] * input_ac[i + 5]);
    input_gate.AccumulateInputForError(i + 6, output_er[i + 6] * input_ac[i + 6]);
    input_gate.AccumulateInputForError(i + 7, output_er[i + 7] * input_ac[i + 7]);
    i += 8;
  }
  for ( ; i < num_cells_; i++) {
   // backpropgate to input
   input.AccumulateInputForError(i, output_er[i] * ig_ac[i]);
   // backpropgate to input gate
   input_gate.AccumulateInputForError(i, output_er[i] * input_ac[i]);
  }
}

void NeuralNetLSTMCell::BackPropagate(const NeuralNetLayerBase &hidden_layer, NeuralNetLayer &input_layer, NeuralNetLayer &input_gate_layer,NeuralNetLayer &output_gate_layer, NeuralNetLayer &forget_gate_layer) {
  assert(input_layer.nneurons() == num_cells_);
  assert(input_gate_layer.nneurons() == num_cells_);
  assert(output_gate_layer.nneurons() == num_cells_);
  assert(forget_gate_layer.nneurons() == num_cells_);
  assert(hidden_layer.nneurons() == num_cells_);
  

  // hidden layer -> forget gate & memory cell output layer
  mc_output_layers_.ResetInputForErrors();
  output_gate_layer.ResetInputForErrors();
  GateBackPropagate(hidden_layer, mc_output_layers_, output_gate_layer);
  mc_output_layers_.ComputeErrors();
  output_gate_layer.ComputeErrors();

  // memory cell output layer -> memory cell layer
  NeuralNetIdentityLayer &current_memory_cell_layer = memory_cell_layers_[0];
  //NeuralNetCenteredSigmoidLayer &current_memory_cell_layer = memory_cell_layers_[0];
  current_memory_cell_layer.ResetInputForErrors();
  //map_connection_mc_mco_.BackPropagate(mc_output_layers_, current_memory_cell_layer);
  map_connection_.BackPropagate(mc_output_layers_, current_memory_cell_layer);

  // output gate -> memory cell
  connection_cell_og_.BackPropagate(output_gate_layer, current_memory_cell_layer);
  current_memory_cell_layer.ComputeErrors();

  // current memory cell -> previous memory cell & forget gate
  memory_cell_layers_[1].ResetInputForErrors();
  forget_gate_layer.ResetInputForErrors();
  GateBackPropagate(current_memory_cell_layer, memory_cell_layers_[1], forget_gate_layer);
  forget_gate_layer.ComputeErrors();
  // current memory cell -> input layer and input gate
  input_layer.ResetInputForErrors();
  input_gate_layer.ResetInputForErrors();
  GateBackPropagate(current_memory_cell_layer, input_layer, input_gate_layer);
  input_layer.ComputeErrors();
  input_gate_layer.ComputeErrors();

  /*
  // input gate layer -> previous memory cell
  connection_cell_ig_.BackPropagate(input_gate_layer_, memory_cell_layers_[1]);
  // forget gate layer -> previous memory cell
  connection_cell_fg_.BackPropagate(output_gate_layer_, memory_cell_layers_[1]);
  memory_cell_layers_[1].ComputeErrors();
  */

  // Accumulate Gradients
  // TODO: ig and fg should be backpropagate with preivous memory cell
  connection_cell_ig_.AccumulateGradients(memory_cell_layers_[1], input_gate_layer);
  connection_cell_fg_.AccumulateGradients(memory_cell_layers_[1], forget_gate_layer);
  connection_cell_og_.AccumulateGradients(output_gate_layer, current_memory_cell_layer);
  
}


void NeuralNetLSTMCell::ReadCell(ifstream &ifs) {
  cout << "============================" << endl;
  cout << "reading LSTM cell" << endl;
  
  neuralnet::read_single(ifs, num_cells_);
  cout << "num_cells_: " << num_cells_ << endl;
  neuralnet::read_single(ifs, bptt_unfold_level_);
  cout << "bptt_unfold_level_: " << bptt_unfold_level_ << endl;
  neuralnet::read_single(ifs, adagrad_);
  cout << "adagrad_: " << adagrad_ << endl;
  neuralnet::read_single(ifs, l2_regularization_param_);
  cout << "l2_regularization_param_: " << l2_regularization_param_ << endl;

  AllocateModel();

  connection_cell_ig_.ReadConnection(ifs);
  connection_cell_og_.ReadConnection(ifs);
  connection_cell_fg_.ReadConnection(ifs);

}

void NeuralNetLSTMCell::WriteCell(ofstream &ofs) {
  neuralnet::write_single(ofs, num_cells_);
  neuralnet::write_single(ofs, bptt_unfold_level_);
  neuralnet::write_single(ofs, adagrad_);
  neuralnet::write_single(ofs, l2_regularization_param_);
  
  connection_cell_ig_.WriteConnection(ofs);
  connection_cell_og_.WriteConnection(ofs);
  connection_cell_fg_.WriteConnection(ofs);

}

void NeuralNetLSTMCell::PrintParams() {
  // TODO
}

void NeuralNetLSTMCell::CheckParams() {
  if (errorinput_cutoff_ < 0) {
    cerr << "errorinput_cutoff should be non-negative!" << endl;
    exit(EXIT_FAILURE);
  }
  if (bptt_unfold_level_ < 1) {
    cerr << "LSTM: BPTT unfold level should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }

  if (num_cells_ < 1) {
    cerr << "LSTM: num_hiddens should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }
}

void NeuralNetLSTMCell::CacheCurrentParams() {
  last_connection_cell_ig_ = connection_cell_ig_;
  last_connection_cell_og_ = connection_cell_og_;
  last_connection_cell_fg_ = connection_cell_fg_;
  // Note: it is not clear whether the memory cells have to be cached as well.
}

void NeuralNetLSTMCell::RestoreLastParams() {
  connection_cell_ig_ = last_connection_cell_ig_;
  connection_cell_og_ = last_connection_cell_og_;
  connection_cell_fg_ = last_connection_cell_fg_;
}

} // namespace nnlm
