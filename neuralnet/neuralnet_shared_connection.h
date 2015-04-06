#ifndef NEURALNET_NEURALNET_SHARED_CONNECTION_H_
#define NEURALNET_NEURALNET_SHARED_CONNECTION_H_

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

// Shared connection.
// If it is row major (storage_input_major_ == true), then the columns for each
// row are shared.
// Otherwise, the rows for each column are shared.
class NeuralNetSharedConnection {

 public:
  // Constructor.
  NeuralNetSharedConnection() : storage_input_major_(false), ninput_(0), noutput_(0),
    l2_regularization_param_(0), adagrad_(false), num_updates_(0) {};
  // Destructor.
  virtual ~NeuralNetSharedConnection() {};

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
  // const std::vector<std::vector<WeightType>>& weights() const { return weights_; }

  // Sets the flag storage_input_major_ and also the dimensions.
  // Automatically calls AllocateConnection().
  void set_dims(bool s, std::size_t ninput, std::size_t noutput) { 
    storage_input_major_ = s;
    ninput_ = ninput;
    noutput_ = noutput;
    AllocateConnection();
  }
  void set_l2_regularization_param(float l2) { l2_regularization_param_ = l2; }
  void set_adagrad(bool adagrad) { adagrad_ = adagrad; }
  void set_weights(std::size_t idx, WeightType val) { 
    assert((idx < ninput_ && storage_input_major_) || (idx < noutput_ && !storage_input_major_));
    weights_[idx] = val;
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
    write_1d_vector(ofs, weights_);
    // TODO: write l2_regularization_param as well
  }
  // Reads the connection from stream.
  void ReadConnection(std::ifstream &ifs) {
    std::cout << "***reading shared connection***" << std::endl;
    read_single(ifs, storage_input_major_);
    std::cout << "storage_input_major_: " << storage_input_major_ << std::endl;
    read_single(ifs, ninput_);
    std::cout << "ninput_: " << ninput_ << std::endl;
    read_single(ifs, noutput_);
    std::cout << "noutput_: " << noutput_ << std::endl;
    read_1d_vector(ifs, weights_);
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

  // Updates the weights in the skipped row/column.
  // idx is the row index when storage_input_major_ is true;
  // and is the column index when storage_input_major_ is false.
  void FastUpdateWeightsMinor(std::size_t idx);

  // True if the weights and gradients are stored in row major form.
  // False if the weights and gradients are stored in column major form.
  bool storage_input_major_;

  // The number of inputs of the connection.
  std::size_t ninput_;
  // The number of outputs of the connection.
  std::size_t noutput_;

  // L2 regularzation parameter.
  float l2_regularization_param_;
  // True if using AdaGrad.
  bool adagrad_;

  // The weights of the connection.
  std::vector<WeightType> weights_;
  // The gradients_ of the connection.
  std::vector<GradientType> gradients_;
  // The learning rate scale for AdaGrad, i.e, sum of gradient squares.
  // Note: it should take into account the regularization as a part of the
  // gradients.
  std::vector<GradientType> sum_gradient_squares_;

  // Random number generator related paraemeters.
  boost::uniform_01<double> uniform_01_;

  // Fast update trick for regularization.
  // The following two parameters (num_updates_ and lastupdate_) are used to
  // accelarate the computation. 
  // If input.activation is 0, temporarily skip updating the weights in the
  // corresponding row, as only the regularization parts need to be updated.
  // When the input.activation is 1, update the weights in the corresponding row
  // before ForwardPropagate.
  // The updated weights only depend on the old weights and number of skipped
  // updates.
  // Total counts of updates.
  int num_updates_;
  // Last update for row/column.
  // It stores the row indices when storage_input_major_ is true;
  // and stores the column indices when storage_input_major_ is false.
  std::vector<int> lastupdate_;
  // Touched rows/columns of gradients_ (weights with nonzero gradient for the
  // non-regularization part).
  // It stores the row indices when storage_input_major_ is true;
  // and stores the column indices when storage_input_major_ is false.
  // FIXME: maybe should use set or unordered_map to avoid the vector goes unbounded
  // when AccumulateGradients are called a lot of times without FastUpdateWeightsMxx (e.g., in BatchSGD).
  std::vector<std::size_t> gradients_touched_;
  // Last learning rate. It is used to ensure the FastUpdateWeightsMinor is
  // called before teh learning rate changes.
  // -1 means FastUpdateWeightsMinor is called last time.
  float last_learning_rate_;
};

} // namespace neuralnet

#endif
