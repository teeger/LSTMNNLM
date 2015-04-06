#ifndef NEURALNET_NEURALNET_CONNECTION_H_
#define NEURALNET_NEURALNET_CONNECTION_H_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "futil.h"
#include "neuralnet_layer_base.h"
#include "neuralnet_layer.h"
#include "neuralnet_sparse_layer.h"

namespace neuralnet {

// [Normal Usage]
// FOR LOOP:
//    ForwardPropagate(input, output); 
//    BackPropagate(output, input); 
//    AccumulateGradients(input, output);
//    if (update) FastUpdateWeightsMajor(learning_rate);
//  FastUpdateWeightsMinor();
//
//  [Fine Control on ForwardPropagate]
//  ForwardPropagateForInput(input, output, i);
//  ForwardPropagateForOutput(input, output, j);
//
//  [Fine Control on BackPropagate && AccumulateGradients]
//  BackPropagateForInput(output, input, i); AccumulateGradientsForInput(input, output, i);
//  BackPropagateForOutput(output, input, j); AccumulateGradientsForOutput(input, output, j);
//
//  [Note]
//  1. For each ForwardPropagate, the BackwardPropagate and the AccumulateGradients
//  must be called before calling another ForwardPropagate.
//  2. If not fine control on the connection methods is needed, then the storage_input_major_
//  should depends on ninput_ and noutput_. If ninput_ > noutput_ (and input is not SparseLayer),
//  then prefer storage_input_major_ = False; otherwise prefer storage_input_major_ = True.
//  3. Currently, large l2_regularization_param_ makes the SGD unstable and slow
//  (don't know exact reason). Needs to check papers about SGD with
//  regularization.
class NeuralNetConnection {

 public:
  // Constructor.
  NeuralNetConnection() : storage_input_major_(true), ninput_(0), noutput_(0),
    l2_regularization_param_(0), adagrad_(false), num_updates_(0) {};
  // Destructor.
  virtual ~NeuralNetConnection() {};

  // Accessors not used.
  // std::size_t nipnut() const { return ninput_; }
  // std::size_t noutput() const { return noutput_; }
  // float l2_regularization_param() const { return l2_regularization_param_; }
  // WeightType weights(std::size_t i, std::size_t j) const { 
  //   assert(i < noutput_);
  //   assert(j < ninput_);
  //   if (transpose_) {
  //     return weights_[j][i]; 
  //   } else {
  //     return weights_[i][j]; 
  //   }
  // }
  // Returns the flag storage_input_major_.
  bool storage_input_major() const { return storage_input_major_; }
  // Returns the weights.
  const std::vector<std::vector<WeightType>>& weights() const { return weights_; }

  // Sets the flag storage_input_major_ and also the dimensions.
  // Automatically calls AllocateConnection().
  void set_dims(bool s, std::size_t ninput, std::size_t noutput) { 
    storage_input_major_ = s;
    ninput_ = ninput;
    noutput_ = noutput;
    AllocateConnection();
  }
  // Sets the l2 regularization parameter.
  void set_l2_regularization_param(float l2) { l2_regularization_param_ = l2; }
  // Sets the flag adagrad_.
  void set_adagrad(bool adagrad) { adagrad_ = adagrad; }
  // Sets the weights.
  void set_weights(std::size_t i, std::size_t j, WeightType val) { 
    assert(i < ninput_);
    assert(j < noutput_);
    if (storage_input_major_) {
      weights_[i][j] = val;
    } else {
      weights_[j][i] = val;
    }
  }

  // Resets the connection.
  // weights_ and gradients_ are reset to 0.
  // sum_gradient_squares_ are reset to 1.
  // num_updates_ and lastupdate_ are reset to 0.
  // last_learning_rate_ is reset  to -1.
  void ResetConnection() {
    size_t dim;
    if (storage_input_major_) {
      dim = ninput_;
    } else {
      dim = noutput_;
    }
    for (size_t d = 0; d < dim; d++) {
      std::fill(weights_[d].begin(), weights_[d].end(), 0.0f);
      std::fill(gradients_[d].begin(), gradients_[d].end(), 0.0f);
      std::fill(sum_gradient_squares_[d].begin(), sum_gradient_squares_[d].end(), 1.0f);
    }
    std::fill(lastupdate_.begin(), lastupdate_.end(), 0);
    gradients_touched_.clear();
    num_updates_ = 0;
    last_learning_rate_ = -1;
  }
  // Randomly initializes weights.
  // TODO: check Ilya Sutskever's blog post about initialization.
  void RandomlyInitialize(boost::mt19937 &rng_engine);
  // Normalizes the weights to ensure the l2-norm of each row/col to be 1.
  // void L2NormalizeWeights(bool by_row);

  // Computes the weighted sum of the activations of input layer neurons and
  // propagates them to the output layer.
  // output.activationinputs = weights_ * input.activations
  void ForwardPropagate(const NeuralNetLayer &input, NeuralNetLayerBase &output);
  void ForwardPropagate(const NeuralNetSparseLayer &input, NeuralNetLayerBase &output);
  // Forward propagates for one input neuron. 
  // Note: Recommend when storage_input_major_ is true. Otherwise it is slow.
  void ForwardPropagateForInput(const NeuralNetLayerBase &input, NeuralNetLayerBase &output, std::size_t idx);
  // Forward propagates for one output neuron. 
  // Note: Recommend when storage_input_major_ is false. Otherwise it is slow.
  void ForwardPropagateForOutput(const NeuralNetLayer &input, NeuralNetLayerBase &output, std::size_t idx);
  void ForwardPropagateForOutput(const NeuralNetSparseLayer &input, NeuralNetLayerBase &output, std::size_t idx);

  // Computes the weighted sum of the error derivatives of the output layer neurons and
  // propagates them to the input layer. 
  // input.errorinputs = weights_.transpose() * output.errors
  void BackPropagate(const NeuralNetLayerBase &output, NeuralNetLayerBase &input);
  // Back propagates for one input neuron. 
  // Note: Recommend when storage_input_major_ is true. Otherwise it is slow.
  void BackPropagateForInput(const NeuralNetLayerBase &output, NeuralNetLayerBase &input, std::size_t idx);
  // Back propagates for one output neuron. 
  // Note: Recommend when storage_input_major_ is false. Otherwise it is slow.
  void BackPropagateForOutput(const NeuralNetLayerBase &output, NeuralNetLayerBase &input, std::size_t idx);

  // Updates the gradients of the connection.
  // gradients_ += input.activations * output.errors 
  //    input.activations:  activations of input layer neurons
  //    output.errors:      errors of output layer neurons
  void AccumulateGradients(const NeuralNetLayer &input, const NeuralNetLayerBase &output);
  void AccumulateGradients(const NeuralNetSparseLayer &input, const NeuralNetLayerBase &output);
  // Updates the gradients for weights corresponding to one input neuron.
  // Note: Recommend when storage_input_major_ is true. Otherwise it is slow.
  void AccumulateGradientsForInput(const NeuralNetLayerBase &input, const NeuralNetLayerBase &output, std::size_t idx);
  // Updates the gradients for weights corresponding to one output neuron.
  // Note: Recommend when storage_input_major_ is false. Otherwise it is slow.
  void AccumulateGradientsForOutput(const NeuralNetLayer &input, const NeuralNetLayerBase &output, std::size_t idx);
  void AccumulateGradientsForOutput(const NeuralNetSparseLayer &input, const NeuralNetLayerBase &output, std::size_t idx);
  
  // Fast updates the weights of the connection.
  // weights_ += learning_rate * (gradients_ - l2_regularization_param_ * weights_);
  // The gradients of the connection are reset automatically.
  // Note: 
  // 1. gradents_ only stores the non-regularized part of the gradients.
  // 2. The update is split into two steps.
  // FastUpdateWeightsMajor updates the weights corresponding to the input
  // neurons with non-zero activations (i.e., gradients_ = 0).
  // FastUpdateWeightsMinor updates the l2_regularization_param_ * weights_
  // part for those skipped weights in FastUpdateWeightsMajor, i.e., those
  // corresponding to the input neurons with zero activations.
  // In ForwardPropagate, when activation becomes 1, FastUpdateWeightsMinor is
  // called for corresponding skipped weights.
  void FastUpdateWeightsMajor(float learning_rate);
  // Updates the weights in all skipped rows/columns.
  // It has to be called before learning rate changes, or evaluating/saving the connections.
  // In current setting, it is called at the end of each Epoch.
  void FastUpdateWeightsMinor();


  // Writes the connection to stream.
  void WriteConnection(std::ofstream &ofs) {
    write_single(ofs, storage_input_major_);
    write_single(ofs, ninput_);
    write_single(ofs, noutput_);
    write_2d_vector(ofs, weights_);
    // TODO: write l2_regularization_param as well
  }
  // Reads the connection from stream.
  void ReadConnection(std::ifstream &ifs) {
    std::cout << "***reading connection***" << std::endl;
    read_single(ifs, storage_input_major_);
    std::cout << "storage_input_major_: " << storage_input_major_ << std::endl;
    read_single(ifs, ninput_);
    std::cout << "ninput_: " << ninput_ << std::endl;
    read_single(ifs, noutput_);
    std::cout << "noutput_: " << noutput_ << std::endl;
    read_2d_vector(ifs, weights_);
    // TODO: read l2_regularization_param as well
  }

  // Writes the connection in txt format to ostream.
  void WriteConnectionToTxt(std::ostream &os);

 private:
  double gen_random_double_01(boost::mt19937 &rng_engine) {
    boost::variate_generator<boost::mt19937&, boost::uniform_01<double> > gen(rng_engine, uniform_01_);
    return gen();
  }

  // Allocates the connection.
  void AllocateConnection();

  // Updates the weights for skipped neurons.
  // idx is the index of input neuron when storage_input_major_ is true;
  // and is the index of output neuron when storage_input_major_ is false.
  void FastUpdateWeightsMinor(std::size_t idx);

  // True if the weights and gradients are stored in input major form.
  // False if the weights and gradients are stored in output major form.
  bool storage_input_major_;

  // The number of inputs of the connection.
  std::size_t ninput_;
  // The number of outputs of the connection.
  std::size_t noutput_;

  // L2 regularization parameter.
  float l2_regularization_param_;
  // True if using AdaGrad.
  bool adagrad_;

  // The weights of the connection.
  std::vector<std::vector<WeightType>> weights_;
  // The gradients_ of the connection.
  std::vector<std::vector<GradientType>> gradients_;
  // The learning rate scale for AdaGrad, i.e, sum of gradient squares.
  // Note: it should take into account the regularization as a part of the
  // gradients.
  std::vector<std::vector<GradientType>> sum_gradient_squares_;

  // Random number generator related parameters.
  boost::uniform_01<double> uniform_01_;

  // Fast update trick for regularization.
  // The following parameters are used to accelerate the computation. 
  // If input.activation is 0, temporarily skip updating the weights in the
  // corresponding row, as only the regularization parts need to be updated.
  // When the input.activation is 1, update the weights in the corresponding row
  // before ForwardPropagate.
  // The updated weights only depend on the old weights and number of skipped
  // updates.
  // Total counts of updates.
  int num_updates_;
  // Last update for the rows/columns of the weights.
  // It stores the input indices when storage_input_major_ is true;
  // and stores the output indices when storage_input_major_ is false.
  std::vector<int> lastupdate_;
  // Touched rows/columns of gradients_ (weights with nonzero gradient for the
  // non-regularization part).
  // It stores the row indices when storage_input_major_ is true;
  // and stores the column indices when storage_input_major_ is false.
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
