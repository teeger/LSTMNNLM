#ifndef FNNLM_MAXENT_FNNLM_H_
#define FNNLM_MAXENT_FNNLM_H_

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <boost/random/mersenne_twister.hpp>
#include "../neuralnet/full_circular_buffer.h"
#include "../neuralnet/neuralnet_types.h"
#include "../neuralnet/neuralnet_connection.h"
#include "../neuralnet/neuralnet_hash_connection.h"
#include "../neuralnet/neuralnet_shared_connection.h"
#include "../neuralnet/neuralnet_sparse_layer.h"
#include "../neuralnet/neuralnet_identity_layer.h"
#include "../neuralnet/neuralnet_softmax_layer.h"
#include "../neuralnet/neuralnet_exp_layer.h"
#include "fnnlm_vocab.h"
#include "fnnlm_data_reader.h"
#include "fnnlm_base.h"

namespace fnnlm {

// Berger's maximum entropy LM (Berger 1996, Mikolov et al. 2011)
// Implemented in Neural Network LM form (though it is not a neural network LM
// but a log-linear LM).
// Available features:
// - ngram feature, encoded in hash table (Mikolov et al. ASRU2011).
class MaxEntFNeuralNetLM : public FNeuralNetLMBase {
 public:
  // Constructor.
  MaxEntFNeuralNetLM() : use_factor_input_(true), use_factor_hidden_(true), 
    ngram_order_(0), hash_table_size_word_(0),
    hash_table_size_mixed_(0), hash_mode_(0) {}
  // Destructor.
  ~MaxEntFNeuralNetLM() {}

  // Sets the flag use_factor_input_.
  void set_use_factor_input(bool f) { use_factor_input_ = f; }
  // Sets the flag use_factor_hidden_.
  void set_use_factor_hidden(bool f) { use_factor_hidden_ = f; }
  // Sets the ngram order.
  void set_ngram_order(std::size_t n) { ngram_order_ = n; }
  // Sets the hash table size for pure-word features.
  void set_hash_table_size_word(std::size_t s) { hash_table_size_word_ = s; }
  // Sets the hash table size for mixed-word-factor features.
  void set_hash_table_size_mixed(std::size_t s) { hash_table_size_mixed_ = s; }
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
  // Maximum entropy neural network parameters
  //===================================
  // True if use the factor (mixed) input layer.
  bool use_factor_input_;
  // True if use the factor hidden layer.
  bool use_factor_hidden_;

  // Ngram order.
  std::size_t ngram_order_;
  // Hash table size for pure-word n-gram features.
  std::size_t hash_table_size_word_;
  // Hash table size for mixed-word-factor n-gram features.
  // It is expected that hash_table_size_mixed_ is much larger 
  // than hash_table_size_word_ since there are more features.
  std::size_t hash_table_size_mixed_;
  // Hash mode.
  // 0 - hash update follows Geoffrey Zweig's implementation (faster but less
  // accurate).
  // 1 - hash update follows Thomas Mikolov's implementation (slower but more
  // accurate).
  int hash_mode_;

  // Context words, i.e., histories.
  // Note: It acts as a layer, so should be processed whenever a layer is
  // processed.
  neuralnet::FullCircularBuffer<std::pair<std::size_t, std::vector<std::size_t>>> context_words_;

  neuralnet::NeuralNetSparseLayer word_input_layer_;
  neuralnet::NeuralNetSparseLayer mixed_input_layer_;
  neuralnet::NeuralNetIdentityLayer factor_hidden_layer_;
  neuralnet::NeuralNetSoftmaxLayer output_layer_;
  neuralnet::NeuralNetExpLayer nce_output_layer_;
  // Since we always need the unigram features, the bias_layer_ is not
  // controlled by the bias_ flag.
  neuralnet::NeuralNetSparseLayer bias_layer_;

  // The word unigram feature weights.
  neuralnet::NeuralNetConnection connection_bias_output_, last_connection_bias_output_;
  // The word n-gram (n>1) feature weights.
  neuralnet::NeuralNetHashConnection connection_wordinput_output_, last_connection_wordinput_output_;
  // Feature weights for mixed-word-factor input + word output
  neuralnet::NeuralNetHashConnection connection_mixedinput_output_, last_connection_mixedinput_output_;
  // The factor unigram feature weights.
  neuralnet::NeuralNetConnection connection_bias_factorhidden_, last_connection_bias_factorhidden_;
  // Feauter weights for pure-word input + factor output. 
  neuralnet::NeuralNetHashConnection connection_wordinput_factorhidden_, last_connection_wordinput_factorhidden_;
  // Feature weights for mixed-word-factor input + factor output.
  neuralnet::NeuralNetHashConnection connection_mixedinput_factorhidden_, last_connection_mixedinput_factorhidden_;

  neuralnet::NeuralNetSharedConnection connection_globalbias_output_, last_connection_globalbias_output_;
  neuralnet::NeuralNetSharedConnection connection_globalbias_factorhidden_, last_connection_globalbias_factorhidden_;

  boost::mt19937 rng_engine_;
};

} // namespace nnlm

#endif
