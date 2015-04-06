#ifndef NNLM_NNLM_BASE_H_
#define NNLM_NNLM_BASE_H_

#include <cstdio>
#include <vector>
#include <string>
#include <climits>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "../neuralnet/neuralnet_types.h"
#include "nnlm_vocab.h"
#include "nnlm_data_reader.h"

namespace nnlm {

// [Training algorithm]: mini-batch stochastic gradient descent
// [Gradient calculation]: back-propagation 
// - maximum likelihood estimation
// - noise constrastive estimation training (Mnih & Teh 2012), extended with
// (?) multiple normalizing constants for context of different sizes.
// [Line search]:
// - vanilla gradient descent
// - AdaGrad (stable for NCE)
// [Stop Criterion]:
// - learning rate is havled at every iteration since the perplexity on the
// validation set increases; the iteration terminates when the perplexity on the
// validation set increases for the second time.
// - (TODO): implement Geoff's log-bilinear LM stop criterion.
class NeuralNetLMBase {
 public:
  // Options related to the training algorithm.
  // - mini-batch stochastic gradient descent
  // - regularization 
  // - line search
  // - noise constrastive estimation
  struct AlgorithmOptions {
    AlgorithmOptions() {
      init_learning_rate_ = 1.0f;
      batch_size_ = 1;
      min_improvement_ = 1.003;
    }

    // Reads the options from the binary ifstream.
    void ReadOpts(std::ifstream &ifs) {
      std::cout << "***reading algorithm options***" << std::endl;
      neuralnet::read_single(ifs, init_learning_rate_);
      std::cout << "init_learning_rate_: " << init_learning_rate_ << std::endl;
      neuralnet::read_single(ifs, batch_size_);
      std::cout << "batch_size_" << batch_size_ << std::endl;
      neuralnet::read_single(ifs, min_improvement_);
      std::cout << "min_improvement_" << min_improvement_ << std::endl;
    }
    // Writes the options into the binary ofstream.
    void WriteOpts(std::ofstream &ofs) {
      neuralnet::write_single(ofs, init_learning_rate_);
      neuralnet::write_single(ofs, batch_size_);
      neuralnet::write_single(ofs, min_improvement_);
    }

    // Initial learning rate.
    float init_learning_rate_;
    // Batch size.
    int batch_size_;
    // Minimum improvement for each iteration. (more details)
    double min_improvement_;
  };

  // Constructor.
  NeuralNetLMBase() : debug_(0), shuffle_datafiles_(false),
    shuffle_sentences_(false), unk_(true), 
    l2_regularization_param_(0.0f), adagrad_(false),
    nce_(false), num_negative_samples_(0), 
    log_num_negative_samples_(-std::numeric_limits<double>::max()), 
    nce_obj_(0),
    independent_(false), globalbias_(false), bias_(false), errorinput_cutoff_(0) {}
  // Destructor.
  virtual ~NeuralNetLMBase() {}

  // Returns the debug level.
  int debug() const { return debug_; }
  // Returns the vocabulary.
  const NNLMVocab& vocab() const { return vocab_; }
  // Returns the l2 regularization parameter.
  float l2_regularization_param() const { return l2_regularization_param_; }
  // Returns the flag adagrad_.
  bool adagrad() const { return adagrad_; }
  // Returns the flag nce_.
  bool nce() const { return nce_; }
  // Returns the number of negative samples.
  std::size_t num_negative_samples() const { return num_negative_samples_; }
  // Returns the log (base e) of the number of negative samples.
  double log_num_negative_samples() const { return log_num_negative_samples_; }
  // Returns the noise_pdf_[w]..
  double noise_pdf(std::size_t w) const { return noise_pdf_[w]; }
  // Returns the nce_obj_.
  double nce_obj() const { return nce_obj_; }
  // Returns the flag globalbias_.
  bool globalbias() const { return globalbias_; }
  // Returns the flag bias_.
  bool bias() const { return bias_; }
  // Returns the errorinput_cutoff_.
  neuralnet::ErrorInputType errorinput_cutoff() const { return errorinput_cutoff_; }

  // Sets the debug level.
  void set_debug(int debug) { debug_ = debug; }
  // Sets the flag unk_.
  void set_unk(bool unk) { unk_ = unk; }
  // Sets the filename of the vocabulary.
  void set_vocab_filename(const std::string &fn) { vocab_filename_ = fn; }
  // Sets the filenames of the training data.
  void set_train_filenames(const std::vector<std::string> &fns) { train_filenames_ = fns; }
  // Sets the flag shuffle_datafiles_.
  void set_shuffle_datafiles(bool s) { shuffle_datafiles_ = s; }
  // Sets the flag shuffle_sentences_.
  void set_shuffle_sentences(bool s) { shuffle_sentences_ = s; }
  // Sets the general options related to the training algorithm.
  void set_algopts(float init_learning_rate, int batch_size, 
                   float min_improvement) {
    algopts_.init_learning_rate_ = init_learning_rate;
    algopts_.batch_size_ = batch_size;
    algopts_.min_improvement_ = min_improvement;
  }
  // Sets the l2 regularaization parameter.
  void set_l2_regularization_param(float l) { l2_regularization_param_ = l; }
  // Sets the flag adagrad_.
  void set_adagrad(bool adagrad) { adagrad_ = adagrad; }
  // Sets the flag nce_.
  void set_nce(bool nce) { nce_ = nce; }
  // Sets the number of negative samples.
  void set_num_negative_samples(std::size_t n) { num_negative_samples_ = n; }
  // Sets the objective value of NCE training.
  void set_nce_obj(double val) { nce_obj_ = val; }
  // Sets the flag independent_.
  void set_independent(bool independent) { independent_ = independent; }
  // Sets the flag globalbias_.
  void set_globalbias(bool gbias) { globalbias_ = gbias; }
  // Sets the flag bias_.
  void set_bias(bool bias) { bias_ = bias; }
  // Sets the cutoff of the input of neuron errors.
  void set_errorinput_cutoff(neuralnet::ErrorInputType c) { errorinput_cutoff_ = c; }

  // Returns a word sampled from the noise distribution.
  std::size_t NCESampleWord(boost::mt19937 &rng_engine);
  // Accumulates the NCE objective value.
  void AccumulateNCEObjetiveValue(double val) { nce_obj_ += val; }

  // Trains the language model.
  void TrainLM(const std::string &validationfile,
               const std::string &outbase,
               bool nce_ppl);
  // Evaluates the perplexity of language model on the txt file.
  void EvalLM(const std::string &infile, bool nce_ppl);

  // Reads the language model from the file.
  void ReadLM(const std::string &infile);
  // Writes the language model to the file.
  void WriteLM(const std::string &outbase);

  // Convert the model binary file to word embedding (input layer) in txt format.
  void ExtractWordInputEmbedding(const std::string &filename);
  // Convert the model binary file to word embedding (output layer) in txt format.
  void ExtractWordOutputEmbedding(const std::string &filename);
  
 private:
  typedef boost::random::discrete_distribution<std::size_t, double> Distribution;

  // Implementation of ReadLM.
  virtual void ReadLMImpl(std::ifstream &ifs) = 0;
  // Implemenation of WriteLM. 
  virtual void WriteLMImpl(std::ofstream &ofs) = 0;
  // Implementation of ExtractWordInputEmbedding.
  virtual void ExtractWordInputEmbeddingImpl(std::ofstream &ofs) = 0;
  // Implementation of ExtractWordOutputEmbedding.
  virtual void ExtractWordOutputEmbeddingImpl(std::ofstream &ofs) = 0;
  // Implementation of CheckParams.
  virtual void CheckParamsImpl() {}
  // Implementation of PrintParams.
  virtual void PrintParamsImpl() {}

  // Allocates the model.
  virtual void AllocateModel() = 0;
  // Initialize the neural network.
  virtual void InitializeNeuralNet() = 0;
  // Resets the neural network activations.
  virtual void ResetActivations() = 0;
  // Caches parameters in current iteration.
  virtual void CacheCurrentParams() = 0;
  // Restores parameters in last iteration.
  virtual void RestoreLastParams() = 0;
  // Forward propagates.
  virtual void ForwardPropagate(std::size_t w) = 0;
  // Back propagates.
  virtual void BackPropagate(std::size_t w) = 0;
  // Updates the connections (fast update trick). 
  virtual void FastUpdateWeightsMajor(float learning_rate) = 0;
  // Updates the connections for skipped rows (fast update trick, i.e., update
  // the regularization part of the gradients).
  // Should be called before learning rate changes, WriteLM or EvalLM.
  virtual void FastUpdateWeightsMinor() = 0;
  // Gets the log-probaility (base e) of the word w.
  // If nce_ is false, then the second parameter is not used.
  // If nce_ is true, then returns the un-normalized log probability.
  virtual double GetLogProb(std::size_t w, bool nce_exact) = 0; 
  
  // Prints parameters.
  void PrintParams();
  // Checks the parameters.
  void CheckParams();
  // Train the LM using mini-batch stochastic gradient descent.
  void BatchSGDTrain(NNLMDataReader &train_data, NNLMDataReader &validation_data, const std::string &outbase, bool nce_ppl);
  // Evaluates the perplexity of language model on the NNLMDataReader object.
  // Returns the log-likelihood of the data.
  double EvalLM(NNLMDataReader &data, bool nce_ppl);
  // Check the noise distribution.
  void NCECheckSampling();

  //===================================
  // General training parameters
  //===================================
  // Debug level.
  int debug_;
  // True if OOV token is a valid token.
  bool unk_;
  // Filename of the vocabulary in txt format.
  std::string vocab_filename_;
  // Filename(s) of the training data.
  std::vector<std::string> train_filenames_;
  // If true, data files will be shuffled.
  bool shuffle_datafiles_;
  // If true, sentences within the file will be shuffled.
  bool shuffle_sentences_;
  // Options related to the training algorithm.
  AlgorithmOptions algopts_;

  // L2 regularization parameter.
  float l2_regularization_param_;
  // True if using AdaGrad update.
  bool adagrad_;

  //===================================
  // NCE training parameters
  //===================================
  // True if using NCE.
  bool nce_;
  // Number of negative samples per history in gradient computation.
  std::size_t num_negative_samples_;
  // Store the log_num_negative_samples_ to tiny speed-up.
  double log_num_negative_samples_;
  // The noise distribution.
  Distribution noise_distribution_;
  // Store the noise_pdf_ for fast retrieval
  // (much faster than call noise_distribution_.param().probabilities()).
  std::vector<double> noise_pdf_;
  // Objective value of NCE training.
  double nce_obj_;

  //===================================
  // General language model parameters
  //===================================
  // True if activations are reset at the begining of every sentence.
  // Must be consistent in train and eval.
  // For some neuralnet LMs (e.g., the log-bilinear), it has to be true.
  bool independent_;
  // True if using global bias at the last layer.
  // For NCE, this is required to get stable results.
  bool globalbias_;
  // True if using bias at the last layer. 
  // It is recommended that globalbias_ is set if bias_ is set.
  // Not used in MaxEntNeuralNetLM.
  bool bias_;
  // Cutoff of the input of neuron errors.
  // If errorinput_cutoff_ > 0, then
  // 1) if the input of a neuron error is larger than errorinput_cutoff_, then use errorinput_cutoff_ instead.
  // 2) if the input of a neuron error is smaller than -errorinput_cutoff_, then use -errorinput_cutoff_ instead.
  neuralnet::ErrorType errorinput_cutoff_;
  // Vocabulary.
  NNLMVocab vocab_;
};

} // namespace nnlm

#endif
