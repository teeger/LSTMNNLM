#ifndef NNLM_MAXENT_NNLM_H_
#define NNLM_MAXENT_NNLM_H_

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <string>
#include <boost/random/mersenne_twister.hpp>
#include "../neuralnet/full_circular_buffer.h"
#include "../neuralnet/neuralnet_types.h"
#include "../neuralnet/neuralnet_connection.h"
#include "../neuralnet/neuralnet_hash_connection.h"
#include "../neuralnet/neuralnet_shared_connection.h"
#include "../neuralnet/neuralnet_sparse_layer.h"
#include "../neuralnet/neuralnet_softmax_layer.h"
#include "../neuralnet/neuralnet_exp_layer.h"
#include "nnlm_vocab.h"
#include "nnlm_data_reader.h"
#include "nnlm_base.h"

namespace nnlm {

// Berger's maximum entropy LM (Berger 1996, Mikolov et al. 2011)
// Implemented in Neural Network LM form (though it is not a neural network LM
// but a log-linear LM).
// Available features:
// - ngram feature, encoded in hash table (Mikolov et al. ASRU2011) (see Notes #2 and #3) 
//
// Notes:
// 1. This implementation is different from the SRILM toolkit implementatoin.
//    - This implementation uses SGD whereas the SRILM toolkit uses L-BFGS,
//    which is faster.
//    - This implementation uses hash tabel to store the ngram feature weights,
//    whereas the SRILM toolkit uses special data structure to store the ngram
//    feature weights.
// 2. In the early maximum entropy LMs, only the occurred n-grams have
//    feature weights. If we consider all n-gram features, then those weights
//    corresponding to the unseen n-grams still can get non-zero weight. To make
//    the model simpler (Occam's razor), simply ignore those features.
// 3. In Mikolov's RNNLM toolkit (see -direct and -direct-order), it stores the
//    n-gram feature weights in a hash table where the hash index is computed by
//    hashing the n-gram word indices. Collisions are simply ignored. In this
//    case, it is actually considering a matrix of n-gram feature weights, i.e.,
//    for each word, it consider the same number of context features. This is
//    different from the early maximum entropy LM where each word can have
//    different number of context features (see Note #2). This implementation
//    follows Mikolov's design.
//
// TODO:
// 1  distant bigram feature (aka. trigger feature) (ref)
// 2. can be trained faster, e.g., using batch GD and pre-compute n-gram count?
class MaxEntNeuralNetLM : public NeuralNetLMBase {
 public:
  // Constructor.
  MaxEntNeuralNetLM() : ngram_order_(0), hash_table_size_(0), hash_mode_(0) {}
  // Destructor.
  ~MaxEntNeuralNetLM() {}

  // Sets the ngram order.
  void set_ngram_order(std::size_t n) { ngram_order_ = n; }
  // Sets the hash table size.
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
  // Maximum entropy neural network parameters
  //===================================
  // Ngram order.
  std::size_t ngram_order_;
  // Hash table size.
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

  neuralnet::NeuralNetSparseLayer input_layer_;
  neuralnet::NeuralNetSoftmaxLayer output_layer_;
  neuralnet::NeuralNetExpLayer nce_output_layer_;
  // Since we always need the unigram features, the bias_layer_ is not
  // controlled by the bias_ flag.
  neuralnet::NeuralNetSparseLayer bias_layer_;

  // The connection_bias_input_output_ stores the unigram feature weights.
  neuralnet::NeuralNetConnection connection_bias_output_, last_connection_bias_output_;
  // The n-gram (n>1) feature weights.
  neuralnet::NeuralNetHashConnection connection_input_output_, last_connection_input_output_;

  neuralnet::NeuralNetSharedConnection connection_globalbias_output_, last_connection_globalbias_output_;

  boost::mt19937 rng_engine_;
};

} // namespace nnlm

#endif
