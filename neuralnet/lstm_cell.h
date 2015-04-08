#ifndef NEURALNET_NEURALNET_LSTM_CELL_H_
#define NEURALNET_NEURALNET_LSTM_CELL_H_

#include <cstdio>
#include <vector>
#include <string>
#include <climits>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "full_circular_buffer.h"
#include "neuralnet_types.h"
#include "neuralnet_layer_base.h"
#include "neuralnet_layer.h"
#include "neuralnet_connection.h"
#include "neuralnet_map_connection.h"
#include "neuralnet_hash_connection.h"
#include "neuralnet_shared_connection.h"
#include "neuralnet_sparse_layer.h"
#include "neuralnet_sigmoid_layer.h"
#include "neuralnet_softmax_layer.h"
#include "neuralnet_exp_layer.h"
#include "neuralnet_tanh_layer.h"
#include "neuralnet_identity_layer.h"

namespace neuralnet{

class NeuralNetLSTMCell {
 public:
  // Constructor.
  NeuralNetLSTMCell() : bptt_unfold_level_(1), num_cells_(0), 
    debug_(0), adagrad_(true), bias_(true), errorinput_cutoff_(0), 
    l2_regularization_param_(0) {}
  // Destructor.
  virtual ~NeuralNetLSTMCell() {}

  // Returns the debug level.
  int debug() const { return debug_; }
  // Returns the flag bias_.
  bool bias() const { return bias_; }
  // Returns the errorinput_cutoff_.
  ErrorInputType errorinput_cutoff() const { return errorinput_cutoff_; }

  // Sets the debug level.
  void set_debug(int debug) { debug_ = debug; }
  // Sets the flag bias_.
  void set_bias(bool bias) { bias_ = bias; }
  // Sets the cutoff of the input of neuron errors.
  void set_errorinput_cutoff(ErrorInputType c) { errorinput_cutoff_ = c; }
  // Sets the bptt_unfold_level_.
  void set_bptt_unfold_level(std::size_t bptt) { bptt_unfold_level_ = bptt; }
  // Sets the number of cells.
  void set_ncells(std::size_t nc) { num_cells_ = nc; }
  // Sets the adagrad_.
  void set_adagrad(bool adagrad) { adagrad_ = adagrad; }
  // Sets the l2_regularization_param_.
  void set_l2_regularization_param(float l2) { l2_regularization_param_ = l2; }

  // Reads LSTM cell.
  void ReadCell(std::ifstream &ifs);
  // Writes LSTM cell.
  void WriteCell(std::ofstream &ofs);

  // Allocates the model.
  void AllocateModel();
  // Initialize the neural network.
  void InitializeCell();
  // Resets the neural network activations.
  void ResetActivations();
  // Caches parameters in current iteration.
  void CacheCurrentParams();
  // Restores parameters in last iteration.
  void RestoreLastParams();
  // Forward propagates.
  void GateForwardPropagate(const NeuralNetLayer &input, const NeuralNetLayer &gate_input, NeuralNetLayerBase &output);
  void ForwardPropagate(NeuralNetLayer &input_layer, NeuralNetLayer &input_gate_layer, NeuralNetLayer &output_gate_layer, NeuralNetLayer &forget_gate_layer, NeuralNetLayerBase &hidden_layer); 
  // Back propagates.
  void GateBackPropagate(const NeuralNetLayerBase &output, NeuralNetLayer &input, NeuralNetLayer &input_gate);
  void BackPropagate(const NeuralNetLayerBase &hidden_layer, NeuralNetLayer &input_layer, NeuralNetLayer &input_gate_layer,NeuralNetLayer &output_gate_layer, NeuralNetLayer &forget_gate_layer); 
  // Updates the connections (fast update trick). 
  void FastUpdateWeightsMajor(float learning_rate);
  void FastUpdateWeightsMinor();
    
  // Prints parameters.
  void PrintParams();
  // Checks the parameters.
  void CheckParams();

 private:

  //==================================
  //  Long short term cell parameters
  //==================================
  std::size_t bptt;
  std::size_t num_cells_;
  // TODO: maybe this should be passed in
  boost::mt19937 rng_engine_;

  neuralnet::FullCircularBuffer<neuralnet::NeuralNetIdentityLayer> memory_cell_layers_;
  //neuralnet::NeuralNetSigmoidLayer input_gate_layer_;
  //neuralnet::NeuralNetSigmoidLayer output_gate_layer_;
  //neuralnet::NeuralNetSigmoidLayer forget_gate_layer_;
  //neuralnet::NeuralNetTanhLayer input_layer_;
  neuralnet::NeuralNetTanhLayer mc_output_layers_;
  //neuralnet::NeuralNetIdentityLayer mc_output_layers_;

  neuralnet::NeuralNetConnection connection_cell_ig_, last_connection_cell_ig_;
  neuralnet::NeuralNetConnection connection_cell_og_, last_connection_cell_og_;
  neuralnet::NeuralNetConnection connection_cell_fg_, last_connection_cell_fg_;

  neuralnet::NeuralNetMapConnection map_connection_;

  //===================================
  // General training parameters
  //===================================
  // Debug level.
  int debug_;

  //===================================
  // General language model parameters
  //===================================
  // True if using global bias at the last layer.
  // For NCE, this is required to get stable results.

  // True if using bias at the last layer. 
  // It is recommended that globalbias_ is set if bias_ is set.
  // Not used in MaxEntNeuralNetLM.
  bool bias_;
  // Cutoff of the input of neuron errors.
  // If errorinput_cutoff_ > 0, then
  // 1) if the input of a neuron error is larger than errorinput_cutoff_, then use errorinput_cutoff_ instead.
  // 2) if the input of a neuron error is smaller than -errorinput_cutoff_, then use -errorinput_cutoff_ instead.
  neuralnet::ErrorType errorinput_cutoff_;
  bool adagrad_;
  float l2_regularization_param_;
  std::size_t bptt_unfold_level_;
};

} // namespace neuralnet 

#endif
