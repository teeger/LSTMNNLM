#ifndef FNNLM_RECURRENT_FNNLM_H_
#define FNNLM_RECURRENT_FNNLM_H_

#include <cstdio>
#include <string>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include "../neuralnet/full_circular_buffer.h"
#include "../neuralnet/neuralnet_types.h"
#include "../neuralnet/neuralnet_connection.h"
#include "../neuralnet/neuralnet_hash_connection.h"
#include "../neuralnet/neuralnet_shared_connection.h"
#include "../neuralnet/neuralnet_sparse_layer.h"
#include "../neuralnet/neuralnet_identity_layer.h"
#include "../neuralnet/neuralnet_sigmoid_layer.h"
#include "../neuralnet/neuralnet_softmax_layer.h"
#include "../neuralnet/neuralnet_exp_layer.h"
#include "fnnlm_vocab.h"
#include "fnnlm_data_reader.h"
#include "fnnlm_base.h"

namespace fnnlm {

// Mikolov's recurrent neural network LM (Mikolov 2010).
class RecurrentFNeuralNetLM : public FNeuralNetLMBase {

 public:
  // Constructors.
  RecurrentFNeuralNetLM() : bptt_unfold_level_(1), num_hiddens_(0), 
    ngram_order_(0), hash_table_size_(0), hash_mode_(0) {}
  // Destructor.
  ~RecurrentFNeuralNetLM() {}

  // Sets the bptt unfold level.
  void set_bptt_unfold_level (std::size_t bptt) { bptt_unfold_level_ = bptt; }
  // Sets the number of hidden neurons.
  void set_nhiddens(std::size_t nh) { num_hiddens_ = nh; }
  // Sets the ngram order.
  void set_ngram_order(std::size_t n) { ngram_order_ = n; }
  // Sets the hash table sie.
  void set_hash_table_size(std::size_t s) { hash_table_size_ = s; }
  // Sets the hash mode.
  void set_hash_mode(int m) { hash_mode_ = m; }

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
  // Recurrent neural network training parameters
  //===================================
  // The unfold level of BPTT algorithm.
  // 1 - only one time step back in time.
  // Note: Should be no less than 1.
  std::size_t bptt_unfold_level_;

  //===================================
  // Recurrent neural network parameters
  //===================================
  // Number of hidden neurons.
  std::size_t num_hiddens_;

  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSparseLayer> word_input_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSparseLayer> factor_input_layers_;
  neuralnet::FullCircularBuffer<neuralnet::NeuralNetSigmoidLayer> hidden_layers_;
  neuralnet::NeuralNetIdentityLayer factor_hidden_layer_;
  neuralnet::NeuralNetSoftmaxLayer output_layer_;
  neuralnet::NeuralNetExpLayer nce_output_layer_;
  neuralnet::NeuralNetSoftmaxLayer factor_output_layer_;
  neuralnet::NeuralNetExpLayer nce_factor_output_layer_;

  neuralnet::NeuralNetConnection connection_wordinput_hidden_, last_connection_wordinput_hidden_;
  neuralnet::NeuralNetConnection connection_factorinput_hidden_, last_connection_factorinput_hidden_;
  neuralnet::NeuralNetConnection connection_recurrenthidden_, last_connection_recurrenthidden_;
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

  //===================================
  // Maximum entropy features
  //===================================
  // Ngram order.
  std::size_t ngram_order_;
  // Hash table size (only valid when hash_ is true).
  std::size_t hash_table_size_;
  // Hash mode.
  // 0 - hash update follows Geoffrey Zweig's implementation (faster but less
  // accurate).
  // 1 - hash update follows Thomas Mikolov's implementation (slower but more
  // accurate).
  int hash_mode_;

  // Context words, i.e., histories.
  // Note: It acts as a layer, so should be processed whenever a layer is
  // processed.
  neuralnet::FullCircularBuffer<std::size_t> context_words_;

  neuralnet::NeuralNetSparseLayer mefeat_ngram_input_layer_;
  // If the max-ent features are used, the bias_layer_ is not controlled by the
  // bias_ flag, since we always need the unigram features.
  // The connection_bias_output_ stores the unigram feature weights.
 
  // The n-gram (n>1) feature weights.
  neuralnet::NeuralNetHashConnection connection_mefeatngraminput_output_, last_connection_mefeatngraminput_output_;
};

} // namespace fnnlm

#endif
