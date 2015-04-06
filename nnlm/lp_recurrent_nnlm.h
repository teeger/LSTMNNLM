#ifndef NNLM_LP_RECURRENT_NNLM_H_
#define NNLM_LP_RECURRENT_NNLM_H_

#include <cstdio>
#include <string>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include "../neuralnet/full_circular_buffer.h"
#include "../neuralnet/neuralnet_types.h"
#include "../neuralnet/neuralnet_connection.h"
#include "../neuralnet/neuralnet_shared_connection.h"
#include "../neuralnet/neuralnet_sparse_layer.h"
#include "../neuralnet/neuralnet_identity_layer.h"
#include "../neuralnet/neuralnet_sigmoid_layer.h"
#include "../neuralnet/neuralnet_softmax_layer.h"
#include "../neuralnet/neuralnet_exp_layer.h"
#include "nnlm_vocab.h"
#include "nnlm_data_reader.h"
#include "nnlm_base.h"

namespace nnlm {

// Mikolov's recurrent neural networl LM with an explicit linear projection
// layer for word embedding.
class LPRecurrentNeuralNetLM : public NeuralNetLMBase {

 public:
  // Constructors.
  LPRecurrentNeuralNetLM() : bptt_unfold_level_(1), num_hiddens_(0) {}
  // Destructor.
  ~LPRecurrentNeuralNetLM() {}

  // Sets the bptt unfold level.
  void set_bptt_unfold_level (std::size_t bptt) { bptt_unfold_level_ = bptt; }
  // Sets the number of hidden neurons.
  void set_nhiddens(std::size_t nh) { num_hiddens_ = nh; }

 private:
  // Implementation of ReadLM.
  virtual void ReadLMImpl(std::ifstream &ifs) override;
  // Implemenation of WriteLM. 
  virtual void WriteLMImpl(std::ofstream &ofs) override;
  // Implementation of ExtractWordInputEmbedding.
  virtual void ExtractWordInputEmbeddingImpl(std::ofstream &ofs) override;
  // Implementation of ExtractWordOutputEmbedding.
  virtual void ExtractWordOutputEmbeddingImpl(std::ofstream &ofs) override;
  // Implementation of CheckParams.
  virtual void CheckParamsImpl() override;
  // Implementation of PrintParams.
  virtual void PrintParamsImpl() override;

  // Allocates the model.
  virtual void AllocateModel() override;
  // Initialize the neural network.
  virtual void InitializeNeuralNet() override;
  // Resets the neural network activations.
  virtual void ResetActivations() override;
  // Caches parameters in current iteration.
  virtual void CacheCurrentParams() override;
  // Restores parameters in last iteration.
  virtual void RestoreLastParams() override;
  // Forward propagates.
  virtual void ForwardPropagate(std::size_t w) override;
  // Back propagates.
  virtual void BackPropagate(std::size_t w) override;
  // Updates the connections for skipped rows (fast update trick).
  virtual void FastUpdateWeightsMajor(float learning_rate) override;
  // Updates the connections for skipped rows (fast update trick).
  // Should be called before learning rate changes, WriteLM or EvalLM.
  virtual void FastUpdateWeightsMinor() override;
  // Gets the log-probaility (base e) of the word w.
  virtual double GetLogProb(std::size_t w, bool nce_exact) override;


  //===================================
  // Recurrent neural network training parameters
  //===================================
  std::size_t bptt_unfold_level_;

  //===================================
  // Recurrent neural network parameters
  //===================================
  // Number of hidden neurons.
  std::size_t num_hiddens_;

  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSparseLayer> input_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetIdentityLayer> projection_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSigmoidLayer> hidden_layers_;
  neuralnet::NeuralNetSoftmaxLayer output_layer_;
  neuralnet::NeuralNetExpLayer nce_output_layer_;

  neuralnet::NeuralNetConnection connection_input_projection_, last_connection_input_projection_;
  neuralnet::NeuralNetConnection connection_projection_hidden_, last_connection_projection_hidden_;
  neuralnet::NeuralNetConnection connection_recurrenthidden_, last_connection_recurrenthidden_;
  neuralnet::NeuralNetConnection connection_hidden_output_, last_connection_hidden_output_;

  neuralnet::NeuralNetSparseLayer bias_layer_;
  neuralnet::NeuralNetSharedConnection connection_globalbias_output_, last_connection_globalbias_output_;
  neuralnet::NeuralNetConnection connection_bias_output_, last_connection_bias_output_;

  boost::mt19937 rng_engine_;
};

} // namespace nnlm

#endif
