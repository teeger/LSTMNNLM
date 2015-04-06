#ifndef NEURALNET_NEURALNET_HASH_CONNECTION_H_
#define NEURALNET_NEURALNET_HASH_CONNECTION_H_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "futil.h"
#include "neuralnet_layer_base.h"
#include "neuralnet_sparse_layer.h"

namespace neuralnet {

// Using a 1d vector to store the 2d weights.
// This follows the design in Mikolov's RNNLM toolkit.
// In current implementation, the weights are initialized as 0.
// If the weights need to be initialized randomly, then need advanced
// processing.
//
// The index of the weight in the 1d vector is obtained by the offsetting the
// index of the input layer by the index of the output layer.
// Presumably, the index of the input layer is the hashed value of some
// "features".
// Also, the size of the 1d vector should be the same as the
// input layer size, i.e., the index of the weight should be moduled by the
// input layer.
// This is different from the NeuralNetSparseConnection, where the weights are
// stored in an unordered_map of vectors.
// Here, it is expected to have lower memory usage, but potentially more
// collision.
// This is specifically designed for maximum entropy features.  So only used
// when input layer is NeuralNetSparseInputLayer.
//
// Notes:
// 1. Do not deal with collisions.
// 2. The weights are initialized as 0s.
// 3. I don't have any better name, though the name NeuralNetHashConnection
// isn't really informative.
class NeuralNetHashConnection {

 public:
  // Constructor.
  NeuralNetHashConnection() : ninput_(0), noutput_(0),
    l2_regularization_param_(0), adagrad_(false) {};
  // Destructor.
  virtual ~NeuralNetHashConnection() {};

  // Sets the dimensions and hash table size.
  // Automatically calls AllocateConnection().
  void set_dims(std::size_t ninput, std::size_t noutput) { 
    ninput_ = ninput;
    noutput_ = noutput;
    AllocateConnection();
  }
  void set_l2_regularization_param(float l2) { l2_regularization_param_ = l2; }
  void set_adagrad(bool adagrad) { adagrad_ = adagrad; }
  void set_weights(std::size_t i, std::size_t j, WeightType val) { 
    assert(i < ninput_);
    assert(j < noutput_);
    weights_[i + j] = val;
  }

  // Resets the connection.
  // weights_ and gradients_ are reset to 0.
  // sum_gradient_squares_ are reset to 1.
  // num_updates_ and lastupdate_ are reset to 0.
  // last_learning_rate_ is reset  to -1.
  void ResetConnection() {
    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(gradients_.begin(), gradients_.end(), 0.0f);
    std::fill(sum_gradient_squares_.begin(), sum_gradient_squares_.end(), 1.0f);
    lastupdate_.clear();
    gradients_touched_.clear();
    num_updates_ = 0;
    last_learning_rate_ = -1;
  }

  // Computes the weighted sum of the activations of input layer neurons and
  // propagates them to the output layer.
  // output.activationinputs = weights_ * input.activations
  void ForwardPropagate(const NeuralNetSparseLayer &input, NeuralNetLayerBase &output);
  // Forward propagates for one output neuron. 
  void ForwardPropagateForOutput(const NeuralNetSparseLayer &input, NeuralNetLayerBase &output, std::size_t idx);

  // Updates the gradients of the connection.
  // gradients_ += input.activations * output.errors 
  //    input.activations:  activations of input layer neurons
  //    output.errors:      errors of output layer neurons
  void AccumulateGradients(const NeuralNetSparseLayer &input, const NeuralNetLayerBase &output);
  // Updates the gradients for weights corresponding to one output neuron.
  void AccumulateGradientsForOutput(const NeuralNetSparseLayer &input, const NeuralNetLayerBase &output, std::size_t idx);
  
  // Fast updates the weights of the connection.
  void FastUpdateWeightsMajor(float learning_rate);
  // Updates the weights in all skipped rows/columns.
  // It has to be called before learning rate changes, or evaluating/saving the connections.
  // In current setting, it is called at the end of each Epoch.
  void FastUpdateWeightsMinor();


  // Writes the connection to stream.
  void WriteConnection(std::ofstream &ofs) {
    write_single(ofs, ninput_);
    write_single(ofs, noutput_);
    write_1d_vector(ofs, weights_);
  }
  // Reads the connection from stream.
  void ReadConnection(std::ifstream &ifs) {
    std::cout << "***reading hash connection***" << std::endl;
    read_single(ifs, ninput_);
    std::cout << "ninput_: " << ninput_ << std::endl;
    read_single(ifs, noutput_);
    std::cout << "noutput_: " << noutput_ << std::endl;
    read_1d_vector(ifs, weights_);
  }

  // Writes the connection in txt format to ostream.
  void WriteConnectionToTxt(std::ostream &os);

 private:
  void update_minor_forward_propagate_atom(const std::size_t j, const ActivationType ac, std::size_t &offset, NeuralNetLayerBase &output) {
    std::unordered_map<std::size_t, int>::iterator mi = lastupdate_.find(offset);
    if (mi == lastupdate_.end()) {
      lastupdate_[offset] = 0;
      mi = lastupdate_.find(offset);
    } 
    if (mi->second < num_updates_) {
      FastUpdateWeightsMinor(offset);
    }

    output.AccumulateInputForActivation(j , ac * weights_[offset++]);
  }

  // Allocates the connection.
  void AllocateConnection();

  // Updates the skipped weights.
  void FastUpdateWeightsMinor(std::size_t idx);

  // The number of inputs of the connection.
  std::size_t ninput_;
  // The number of outputs of the connection.
  std::size_t noutput_;

  // L2 regularization parameter.
  float l2_regularization_param_;
  // True if using AdaGrad.
  bool adagrad_;

  // The weights of the connection.
  std::vector<WeightType> weights_;
  // The gradients_ of the connection.
  std::vector<GradientType> gradients_;
  // The learning rate scale for AdaGrad, i.e, sum of gradient squares.
  std::vector<GradientType> sum_gradient_squares_;

  // Fast update trick for regularization.
  // The following parameters are used to accelerate the computation. 
  // Total counts of updates.
  int num_updates_;
  // Last update for the weights.
  std::unordered_map<std::size_t, int> lastupdate_;
  // Indices of touched gradients_ (weights with nonzero gradient for the
  // non-regularization part).
  // FIXME: maybe should use set or unordered_map/set to avoid the vector goes unbounded
  // when AccumulateGradients are called a lot of times without FastUpdateWeightsMxx (e.g., in BatchSGD).
  std::vector<std::size_t> gradients_touched_;
  // Last learning rate. It is used to ensure the FastUpdateWeightsMinor is
  // called before the learning rate changes.
  // -1 means FastUpdateWeightsMinor is called last time.
  float last_learning_rate_;
};

} // namespace neuralnet

#endif
