#ifndef FNNLM_LOGBILINEAR_FNNLM_H_
#define FNNLM_LOGBILINEAR_FNNLM_H_

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
#include "../neuralnet/neuralnet_softmax_layer.h"
#include "../neuralnet/neuralnet_exp_layer.h"
#include "fnnlm_vocab.h"
#include "fnnlm_data_reader.h"
#include "fnnlm_base.h"

namespace fnnlm {

// Botha and Blunsom's log-bilinear LM++ (Botha and Blunsom 2014).
// LBL++: use_factor_input_ == true && use_factor_hidden_ == true
// LBL+c: use_factor_input_ == true && use_factor_hidden_ == false
// LBL+o: use_factor_input_ == false && use_factor_hidden_ == true
// LBL: use_factor_input_ == false && use_factor_hidden_ == false (recommend to
// use logbilinear_nnlm instead since it has less if-statement)
class LogBilinearFNeuralNetLM : public FNeuralNetLMBase {
 public:
  // Constructor.
  LogBilinearFNeuralNetLM() : context_size_(0), num_hiddens_(0) {}
  // Destructor.
  ~LogBilinearFNeuralNetLM() {}

  // Sets the context size.
  void set_context_size(std::size_t cs) { context_size_ = cs; }
  // Sets number of hidden neurons / dimension of the word feature vector.
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
  virtual void ForwardPropagate(std::size_t w, const std::vector<std::size_t> &fs) override;
  // Back propagates.
  virtual void BackPropagate(std::size_t w, const std::vector<std::size_t> &fs) override;
  // Updates the connections for skipped rows (fast update trick).
  virtual void FastUpdateWeightsMajor(float learning_rate) override;
  // Updates the connections for skipped rows (fast update trick).
  // Should be called before learning rate changes, WriteLM or EvalLM.
  virtual void FastUpdateWeightsMinor() override;
  // Gets the log-probaility (base e) of the word w.
  virtual double GetLogProb(std::size_t w, bool nce_exact) override;

  //===================================
  // Log-bilinear neural network parameters
  //===================================
  // Size of the context window (= ngram_order - 1).
  std::size_t context_size_;
  // Number of hidden neurons.
  // It is also the dimension of the word feature vectors.
  std::size_t num_hiddens_;

  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSparseLayer> word_input_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSparseLayer> factor_input_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetIdentityLayer> projection_layers_;
  neuralnet::NeuralNetIdentityLayer hidden_layer_;
  neuralnet::NeuralNetIdentityLayer factor_hidden_layer_;
  neuralnet::NeuralNetSoftmaxLayer output_layer_;
  neuralnet::NeuralNetExpLayer nce_output_layer_;
  neuralnet::NeuralNetSoftmaxLayer factor_output_layer_;
  neuralnet::NeuralNetExpLayer nce_factor_output_layer_;

  neuralnet::NeuralNetConnection connection_wordinput_projection_, last_connection_wordinput_projection_;
  neuralnet::NeuralNetConnection connection_factorinput_projection_, last_connection_factorinput_projection_;
  std::vector<neuralnet::NeuralNetConnection> connections_projection_hidden_, last_connections_projection_hidden_;
  neuralnet::NeuralNetConnection connection_hidden_output_, last_connection_hidden_output_;
  neuralnet::NeuralNetConnection connection_hidden_factorhidden_, last_connection_hidden_factorhidden_;
  neuralnet::NeuralNetConnection connection_hidden_factoroutput_, last_connection_hidden_factoroutput_;

  neuralnet::NeuralNetSparseLayer bias_layer_;
  neuralnet::NeuralNetSharedConnection connection_globalbias_output_, last_connection_globalbias_output_;
  neuralnet::NeuralNetConnection connection_bias_output_, last_connection_bias_output_;
  neuralnet::NeuralNetSharedConnection connection_globalbias_factorhidden_, last_connection_globalbias_factorhidden_;
  neuralnet::NeuralNetConnection connection_bias_factorhidden_, last_connection_bias_factorhidden_;
  neuralnet::NeuralNetSharedConnection connection_globalbias_factoroutput_, last_connection_globalbias_factoroutput_;
  neuralnet::NeuralNetConnection connection_bias_factoroutput_, last_connection_bias_factoroutput_;

  boost::mt19937 rng_engine_;
};

} // namespace fnnlm

#endif
