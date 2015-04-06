#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include "../neuralnet/futil.h"
#include "../neuralnet/neuralnet_numeric.h"
#include "recurrent_nnlm.h"

// <cstdio>
using std::size_t;
// <cstdlib>
using std::exit;
// <cmath>
using std::log;
using std::exp;
// <ctime>
using std::clock_t;
using std::clock;
// <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
// <fstream>
using std::ifstream;
using std::ofstream;
// <ios>
using std::ios;
// <string>
using std::string;
// <vector>
using std::vector;
// <unordered_map>
using std::unordered_map;
// <algorithm>
using std::sort;
using std::unique;
// <limits>
using std::numeric_limits;

namespace nnlm {

void RecurrentNeuralNetLM::ReadLMImpl(ifstream &ifs) {
  neuralnet::read_single(ifs, num_hiddens_);
  cout << "num_hiddens_: " << num_hiddens_ << endl;

  neuralnet::read_single(ifs, ngram_order_);
  cout << "ngram_order_: " << ngram_order_ << endl;
  neuralnet::read_single(ifs, hash_table_size_);
  cout << "hash_table_size_: " << hash_table_size_ << endl;
  neuralnet::read_single(ifs, hash_mode_);
  cout << "hash_mode_: " << hash_mode_ << endl;

  AllocateModel();

  connection_input_hidden_.ReadConnection(ifs);
  connection_recurrenthidden_.ReadConnection(ifs);
  connection_hidden_output_.ReadConnection(ifs);
  
  if (globalbias()) {
    connection_globalbias_output_.ReadConnection(ifs);
  }
  if (bias()) {
    connection_bias_output_.ReadConnection(ifs);
  }

  if (ngram_order_ > 1) {
    connection_mefeatngraminput_output_.ReadConnection(ifs);
  }
}

void RecurrentNeuralNetLM::WriteLMImpl(ofstream &ofs) {
  neuralnet::write_single(ofs, num_hiddens_);

  neuralnet::write_single(ofs, ngram_order_);
  neuralnet::write_single(ofs, hash_table_size_);
  neuralnet::write_single(ofs, hash_mode_);

  connection_input_hidden_.WriteConnection(ofs);
  connection_recurrenthidden_.WriteConnection(ofs);
  connection_hidden_output_.WriteConnection(ofs);

  if (globalbias()) {
    connection_globalbias_output_.WriteConnection(ofs);
  }
  if (bias()) {
    connection_bias_output_.WriteConnection(ofs);
  }

  if (ngram_order_ > 1) {
    connection_mefeatngraminput_output_.WriteConnection(ofs);
  }
}

void RecurrentNeuralNetLM::ExtractWordInputEmbeddingImpl(ofstream &ofs) {
  const vector<vector<neuralnet::WeightType>> &weights = connection_input_hidden_.weights();
  if (connection_input_hidden_.storage_input_major()) {
    assert(weights.size() == vocab().size());
    for (size_t i = 0; i < vocab().size(); i++) {
      ofs << vocab().word4idx(i);
      const vector<neuralnet::WeightType> &wt = weights[i];
      assert(wt.size() == num_hiddens_);
      for (size_t j = 0; j < num_hiddens_; j++) {
        ofs << " " << wt[j];
      }
      ofs << endl;
    }
  } else {
    assert(weights.size() == num_hiddens_);
    for (size_t i = 0; i < vocab().size(); i++) {
      ofs << vocab().word4idx(i);
      for (size_t j = 0; j < num_hiddens_; j++) {
        assert(weights[j].size() == vocab().size());
        ofs << " " << weights[j][i];
      }
      ofs << endl;
    }
  }
}

void RecurrentNeuralNetLM::ExtractWordOutputEmbeddingImpl(ofstream &ofs) {
  const vector<vector<neuralnet::WeightType>> &weights = connection_hidden_output_.weights();
  if (connection_hidden_output_.storage_input_major()) {
    assert(weights.size() == num_hiddens_);
    for (size_t i = 0; i < vocab().size(); i++) {
      ofs << vocab().word4idx(i);
      for (size_t j = 0; j < num_hiddens_; j++) {
        assert(weights[j].size() == vocab().size());
        ofs << " " << weights[j][i];
      }
      ofs << endl;
    }
  } else {
    assert(weights.size() == vocab().size());
    for (size_t i = 0; i < vocab().size(); i++) {
      ofs << vocab().word4idx(i);
      const vector<neuralnet::WeightType> &wt = weights[i];
      assert(wt.size() == num_hiddens_);
      for (size_t j = 0; j < num_hiddens_; j++) {
        ofs << " " << wt[j];
      }
      ofs << endl;
    }
  }
}

void RecurrentNeuralNetLM::PrintParamsImpl() {
  cout << "num_hiddens_: " << num_hiddens_ << endl;

  cout << "ngram_order_: " << ngram_order_ << endl;
  cout << "hash_table_size_: " << hash_table_size_ << endl;
  cout << "hash_mode_: " << hash_mode_ << endl;
}

void RecurrentNeuralNetLM::CheckParamsImpl() {
  if (bptt_unfold_level_ < 1) {
    cerr << "BPTT unfold level should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }

  if (num_hiddens_ < 1) {
    cerr << "num_hiddens should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }

  if (ngram_order_ > 0) {
    if (!bias()) {
      cerr << "bias_ should be true (unigram max-ent feature) when ngram_order_ > 0!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  if (ngram_order_ > 1 && hash_table_size_ < 1) {
    cerr << "hash table size should be no less than 1 when ngram_order_ > 1!" << endl;
    exit(EXIT_FAILURE);
  }
  if (hash_mode_ != 0 && hash_mode_ != 1) {
    cerr << "hash mode can only be 0 or 1 in current implementation!" << endl;
    exit(EXIT_FAILURE);
  }
}

void RecurrentNeuralNetLM::AllocateModel() {
  input_layers_.set_capacity(bptt_unfold_level_);
  for (size_t i = 0; i < bptt_unfold_level_; i++) {
    input_layers_[i].set_nneurons(vocab().size(), true, false);
  }

  hidden_layers_.set_capacity(bptt_unfold_level_ + 1);
  for (size_t i = 0; i < bptt_unfold_level_ + 1; i++) {
    hidden_layers_[i].set_nneurons(num_hiddens_, false, false);
    hidden_layers_[i].set_errorinput_cutoff(errorinput_cutoff());
  }

  if (nce()) {
    nce_output_layer_.set_nneurons(vocab().size(), false, true);
  } else {
    output_layer_.set_nneurons(vocab().size(), false, true);
  }

  connection_input_hidden_.set_dims(true, vocab().size(), num_hiddens_);
  last_connection_input_hidden_.set_dims(true, vocab().size(), num_hiddens_);
  connection_input_hidden_.set_adagrad(adagrad());
  last_connection_input_hidden_.set_adagrad(adagrad());
  connection_input_hidden_.set_l2_regularization_param(l2_regularization_param());
  last_connection_input_hidden_.set_l2_regularization_param(l2_regularization_param());

  connection_recurrenthidden_.set_dims(true, num_hiddens_, num_hiddens_);
  last_connection_recurrenthidden_.set_dims(true, num_hiddens_, num_hiddens_);
  connection_recurrenthidden_.set_adagrad(adagrad());
  last_connection_recurrenthidden_.set_adagrad(adagrad());
  connection_recurrenthidden_.set_l2_regularization_param(l2_regularization_param());
  last_connection_recurrenthidden_.set_l2_regularization_param(l2_regularization_param());

  if (nce()) {
    connection_hidden_output_.set_dims(false, num_hiddens_, vocab().size());
    last_connection_hidden_output_.set_dims(false, num_hiddens_, vocab().size());
  } else {
    // N.B. usually num_hiddens_ is smaller than the size of the output layer.
    // Otherwise, it might be better to use storage_input_major_ = false.
    connection_hidden_output_.set_dims(true, num_hiddens_, vocab().size());
    last_connection_hidden_output_.set_dims(true, num_hiddens_, vocab().size());
  }
  connection_hidden_output_.set_adagrad(adagrad());
  last_connection_hidden_output_.set_adagrad(adagrad());
  connection_hidden_output_.set_l2_regularization_param(l2_regularization_param());
  last_connection_hidden_output_.set_l2_regularization_param(l2_regularization_param());

  if (globalbias() || bias()) {
    // Remember to set activation to 1.
    bias_layer_.set_nneurons(1, true, false);
    bias_layer_.SetActivationsToValue(1.0);
  }
  if (globalbias()) {
    connection_globalbias_output_.set_dims(true, 1, vocab().size());
    last_connection_globalbias_output_.set_dims(true, 1, vocab().size());
    connection_globalbias_output_.set_adagrad(adagrad());
    last_connection_globalbias_output_.set_adagrad(adagrad());
    connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
  }
  if (bias()) {
    if (nce()) {
      connection_bias_output_.set_dims(false, 1, vocab().size());
      last_connection_bias_output_.set_dims(false, 1, vocab().size());
    } else {
      connection_bias_output_.set_dims(true, 1, vocab().size());
      last_connection_bias_output_.set_dims(true, 1, vocab().size());
    }
    connection_bias_output_.set_adagrad(adagrad());
    last_connection_bias_output_.set_adagrad(adagrad());
    connection_bias_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_bias_output_.set_l2_regularization_param(l2_regularization_param());
  }

  // Allocates model parameters related to max-ent features.
  if (ngram_order_ > 1) {
    context_words_.set_capacity(ngram_order_ - 1);
    mefeat_ngram_input_layer_.set_nneurons(hash_table_size_, true, false);

    connection_mefeatngraminput_output_.set_dims(hash_table_size_, vocab().size());
    last_connection_mefeatngraminput_output_.set_dims(hash_table_size_, vocab().size());
    connection_mefeatngraminput_output_.set_adagrad(adagrad());
    last_connection_mefeatngraminput_output_.set_adagrad(adagrad());
    connection_mefeatngraminput_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_mefeatngraminput_output_.set_l2_regularization_param(l2_regularization_param());
  }
}

void RecurrentNeuralNetLM::InitializeNeuralNet() {
  // Reset activations.
  ResetActivations();

  // Randomly initialize connections.
  connection_input_hidden_.RandomlyInitialize(rng_engine_);
  connection_recurrenthidden_.RandomlyInitialize(rng_engine_);
  connection_hidden_output_.RandomlyInitialize(rng_engine_);

  // The connection_bias_output_ and connection_globalbias_output_ are initialized as 0s.

  // The connections for max-ent features are initialized as 0s.
  // Note: at least connection_input_output_ is initialized as all 0s.
}

void RecurrentNeuralNetLM::ResetActivations() {
  // Initialize input layer activations with eos.
  for (size_t i = 0; i < bptt_unfold_level_; i++) {
    input_layers_[i].SetActivationsToValue(0.0f);
    input_layers_[i].set_activations(vocab().eos_idx(), 1.0f);
  }

  // Initialize the latest hidden layer activations with 0.1 to prevent unstability.
  hidden_layers_[0].SetActivationsToValue(0.1f);
  // Initialize all history hidden layer activations with zeros.
  for (size_t i = 1; i < bptt_unfold_level_ + 1; i++) {
    hidden_layers_[i].SetActivationsToValue(0.0f);
  }

  // Initialize the ouptut layer activations with zeros.
  if (nce()) {
    nce_output_layer_.SetActivationsToValue(0.0f);
  } else {
    output_layer_.SetActivationsToValue(0.0f);
  }

  if (ngram_order_ > 1) {
    context_words_.set_offset(0);
    for (size_t i = 0; i < ngram_order_ - 1; i++) {
      context_words_[i] = vocab().eos_idx();
    }
    mefeat_ngram_input_layer_.SetActivationsToValue(0.0f);
  }
}

void RecurrentNeuralNetLM::CacheCurrentParams() {
  last_connection_input_hidden_ = connection_input_hidden_;
  last_connection_recurrenthidden_ = connection_recurrenthidden_;
  last_connection_hidden_output_ = connection_hidden_output_;

  if (globalbias()) {
    last_connection_globalbias_output_ = connection_globalbias_output_;
  }
  if (bias()) {
    last_connection_bias_output_ = connection_bias_output_;
  }

  if (ngram_order_ > 1) {
    last_connection_mefeatngraminput_output_ = connection_mefeatngraminput_output_;
  }
}

void RecurrentNeuralNetLM::RestoreLastParams() {
  connection_input_hidden_ = last_connection_input_hidden_;
  connection_recurrenthidden_ = last_connection_recurrenthidden_;
  connection_hidden_output_ = last_connection_hidden_output_;

  if (globalbias()) {
    connection_globalbias_output_ = last_connection_globalbias_output_;
  }
  if (bias()) {
    connection_bias_output_ = last_connection_bias_output_;
  }

  if (ngram_order_ > 1) {
    connection_mefeatngraminput_output_ = last_connection_mefeatngraminput_output_;
  }
}

void RecurrentNeuralNetLM::ForwardPropagate(size_t w) {  
  // Forward propagation no need to unfold.

  // set input layer activation
  input_layers_.rotate(bptt_unfold_level_ - 1);
  neuralnet::NeuralNetSparseLayer &current_input_layer = input_layers_[0];
  current_input_layer.SetActivationsToValue(0);
  current_input_layer.set_activations(w, 1);

  if (ngram_order_ > 1) {
    context_words_.rotate(ngram_order_ - 2);
    context_words_[0] = w;
    mefeat_ngram_input_layer_.SetActivationsToValue(0);
    // set n-gram max-ent features
    if (hash_mode_ == 0) {
      size_t hv = 5381;
      for (size_t i = 0; i < ngram_order_ - 1; i++) {
        neuralnet::hash_update0(hv, context_words_[i]);
        size_t hidx = hv % hash_table_size_;
        mefeat_ngram_input_layer_.AccumulateActivation(hidx, 1);
      }
    } else if (hash_mode_ == 1) {
      for (size_t a = 1; a < ngram_order_; a++) {
        size_t hidx = neuralnet::HASH_OFFSET;
        for (size_t b = 1; b <= a; b++) {
          neuralnet::hash_update1(hidx, a, b, context_words_[b - 1]);
        }
        hidx = hidx % hash_table_size_;
        mefeat_ngram_input_layer_.AccumulateActivation(hidx, 1);
      }
    } else {
      cerr << "Unknown hash mode!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  // propagate input -> hidden and last_hidden -> hidden
  hidden_layers_.rotate(bptt_unfold_level_);
  neuralnet::NeuralNetSigmoidLayer &current_hidden_layer = hidden_layers_[0];
  current_hidden_layer.ResetInputForActivations();
  connection_recurrenthidden_.ForwardPropagate(hidden_layers_[1], current_hidden_layer);
  connection_input_hidden_.ForwardPropagate(current_input_layer, current_hidden_layer);
  current_hidden_layer.ComputeActivations();

  
  if (!nce()) {
    // propagate hidden -> output
    output_layer_.ResetInputForActivations();
    connection_hidden_output_.ForwardPropagate(current_hidden_layer, output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.ForwardPropagate(bias_layer_, output_layer_);
    }
    if (bias()) {
      connection_bias_output_.ForwardPropagate(bias_layer_, output_layer_);
    }

    // propagate mefeat_ngram_input -> output
    if (ngram_order_ > 1) {
      connection_mefeatngraminput_output_.ForwardPropagate(mefeat_ngram_input_layer_, output_layer_);
    }

    output_layer_.ComputeActivations();
  }
  // If using NCE, do not forward propagate to nce_output_layer_ to save
  // computation.
  // Should call ResetInputForActivation and ForwardPropagateForOutput in
  // BackPropagate or other places as requested.
}

void RecurrentNeuralNetLM::BackPropagate(size_t w) {
  if (nce()) {
    unordered_map<size_t, int> nce_sampled_words;
    // N.B. the positive sample has count 0 initially.
    nce_sampled_words[w] = 0;
    for (size_t i = 0; i < num_negative_samples(); i++) {
      const size_t x = NCESampleWord(rng_engine_);
      unordered_map<size_t, int>::iterator it = nce_sampled_words.find(x);
      if (it != nce_sampled_words.end()) {
        it->second += 1;
      } else {
        nce_sampled_words[x] = 1;
      }
    }

    neuralnet::NeuralNetSigmoidLayer &current_hidden_layer = hidden_layers_[0];
    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
         it != nce_sampled_words.end(); ++it) {
      nce_output_layer_.ResetInputForActivation(it->first);
      connection_hidden_output_.ForwardPropagateForOutput(current_hidden_layer, nce_output_layer_, it->first);
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
      if (bias()) {
        connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
      }
      if (ngram_order_ > 1) {
        connection_mefeatngraminput_output_.ForwardPropagateForOutput(mefeat_ngram_input_layer_, nce_output_layer_, it->first);
      }
    }

    // N.B.: directly set errors to avoid SetErrorsToValue.
    // But should be very careful about postive sample.
    double log_pw, log_kpnw, log_den, rat;
    double obj_val = 0;
    for (unordered_map<size_t, int>::const_iterator it =
         nce_sampled_words.begin(); it != nce_sampled_words.end(); ++it) {
      log_pw = nce_output_layer_.activationinputs(it->first);
      // for numerical stability
      if (log_pw > 50) {
        log_pw = 50;
      }
      if (log_pw < -50) {
        log_pw = -50;
      }
      log_kpnw = log_num_negative_samples() + log(noise_pdf(it->first));
      log_den = log_kpnw;
      neuralnet::logadd(log_den, log_pw);
      rat = exp(log_pw - log_den);
      // assertion for numerical stability
      assert(rat >= 0 && rat <= 1.00001);
      // N.B. nce_sampled_words[w] = 0 initially, and thus, here is either reset
      // the errors to 0, or set the error as a negative sample.
      nce_output_layer_.set_errors(it->first, -rat * it->second);
      obj_val += (log_kpnw - log_den) * it->second;
    }
    log_pw = nce_output_layer_.activationinputs(w);
    // for numerical stability
    if (log_pw > 50) {
      log_pw = 50;
    }
    if (log_pw < -50) {
      log_pw = -50;
    }
    log_kpnw = log_num_negative_samples() + log(noise_pdf(w));
    log_den = log_kpnw;
    neuralnet::logadd(log_den, log_pw);
    rat = exp(log_kpnw - log_den);
    // assertion for numerical stability
    assert(rat >= 0 && rat <= 1.00001);
    nce_output_layer_.AccumulateError(w, rat);
    obj_val += log_pw - log_den;

    AccumulateNCEObjetiveValue(obj_val);

    current_hidden_layer.ResetInputForErrors();
    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
         it != nce_sampled_words.end(); ++it) {
      connection_hidden_output_.BackPropagateForOutput(nce_output_layer_, current_hidden_layer, it->first);
    }
    current_hidden_layer.ComputeErrors();
    // no need to back-propagate error to the oldest hidden layers
    for (size_t i = 1; i < bptt_unfold_level_; i++) {
      hidden_layers_[i].ResetInputForErrors();
      connection_recurrenthidden_.BackPropagate(hidden_layers_[i-1], hidden_layers_[i]);
      hidden_layers_[i].ComputeErrors();
    }

    // no need to back-propagate error to input layers
    // no need to back-propagate error to bias layers

    // globalbias_ must be set when nce_ is true.
    assert(globalbias());
    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
         it != nce_sampled_words.end(); ++it) {
      connection_hidden_output_.AccumulateGradientsForOutput(current_hidden_layer, nce_output_layer_, it->first);
      connection_globalbias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
      if (bias()) {
        connection_bias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
      }
      if (ngram_order_ > 1) {
        connection_mefeatngraminput_output_.AccumulateGradientsForOutput(mefeat_ngram_input_layer_, nce_output_layer_, it->first);
      }
    }
  } else {
    // For the output layer, directly assign errors
    const size_t n = vocab().size();
    size_t i;
    for (i = 0; i + 7 < n; ) {
      output_layer_.set_errors(i, -output_layer_.activations(i));
      output_layer_.set_errors(i + 1, -output_layer_.activations(i + 1));
      output_layer_.set_errors(i + 2, -output_layer_.activations(i + 2));
      output_layer_.set_errors(i + 3, -output_layer_.activations(i + 3));
      output_layer_.set_errors(i + 4, -output_layer_.activations(i + 4));
      output_layer_.set_errors(i + 5, -output_layer_.activations(i + 5));
      output_layer_.set_errors(i + 6, -output_layer_.activations(i + 6));
      output_layer_.set_errors(i + 7, -output_layer_.activations(i + 7));
      i += 8;
    }
    for (; i < n; i++) {
      output_layer_.set_errors(i, -output_layer_.activations(i));
    }
    output_layer_.AccumulateError(w, 1);

    // For hidden layers, first reset input for errors.
    neuralnet::NeuralNetSigmoidLayer &current_hidden_layer = hidden_layers_[0];
    current_hidden_layer.ResetInputForErrors();
    connection_hidden_output_.BackPropagate(output_layer_, current_hidden_layer);
    current_hidden_layer.ComputeErrors();
    // no need to back-propagate error to the oldest hidden layers
    for (size_t i = 1; i < bptt_unfold_level_; i++) {
      hidden_layers_[i].ResetInputForErrors();
      connection_recurrenthidden_.BackPropagate(hidden_layers_[i-1], hidden_layers_[i]);
      hidden_layers_[i].ComputeErrors();
    }

    // no need to back-propagate error to input layers
    // no need to back-propagate error to bias layer
    // no need to back-propagate error to mefeat_ngram input layer

    // Accumulate gradients for connections.
    connection_hidden_output_.AccumulateGradients(current_hidden_layer,
                                                  output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.AccumulateGradients(bias_layer_,
                                                        output_layer_);
    }
    if (bias()) {
      connection_bias_output_.AccumulateGradients(bias_layer_,
                                                  output_layer_);
    }

    if (ngram_order_ > 1) {
      connection_mefeatngraminput_output_.AccumulateGradients(mefeat_ngram_input_layer_, output_layer_);
    }
  }

  for (size_t i = 0; i < bptt_unfold_level_; i++) {
    connection_recurrenthidden_.AccumulateGradients(hidden_layers_[i+1],
                                                    hidden_layers_[i]);
  }
  for (size_t i = 0; i < bptt_unfold_level_; i++) {
    connection_input_hidden_.AccumulateGradients(input_layers_[i],
                                                 hidden_layers_[i]);
  }
}

void RecurrentNeuralNetLM::FastUpdateWeightsMajor(float learning_rate) {
  // Note: when using NCE, vanilla gradient descent does not works (numerically unstable).
  // Adagrad is more stable (and learning_rate = 1.0 is reasonable for lots of scenarios).

  // TODO: update nce params if nce constants are modelled.
  connection_hidden_output_.FastUpdateWeightsMajor(learning_rate);
  connection_recurrenthidden_.FastUpdateWeightsMajor(learning_rate);
  connection_input_hidden_.FastUpdateWeightsMajor(learning_rate);

  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMajor(learning_rate);
  }
  if (bias()) {
    connection_bias_output_.FastUpdateWeightsMajor(learning_rate);
  }

  if (ngram_order_ > 1) {
    connection_mefeatngraminput_output_.FastUpdateWeightsMajor(learning_rate);
  }
}

void RecurrentNeuralNetLM::FastUpdateWeightsMinor() {
  connection_hidden_output_.FastUpdateWeightsMinor();
  connection_recurrenthidden_.FastUpdateWeightsMinor();
  connection_input_hidden_.FastUpdateWeightsMinor();

  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMinor();
  }
  if (bias()) {
    connection_bias_output_.FastUpdateWeightsMinor();
  }

  if (ngram_order_ > 1) {
    connection_mefeatngraminput_output_.FastUpdateWeightsMinor();
  }
}

double RecurrentNeuralNetLM::GetLogProb(size_t w, bool nce_exact) {
  if (!nce()) {
    // Ignore nce_exact.
    return log(output_layer_.activations(w));
  } else {
    // globalbias_ must be set when nce_ is true.
    assert(globalbias());
    double logp;
    if (nce_exact) {
      nce_output_layer_.ResetInputForActivations();
      connection_hidden_output_.ForwardPropagate(hidden_layers_[0], nce_output_layer_);
      connection_globalbias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
      if (bias()) {
        connection_bias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
      }
      if (ngram_order_ > 1) {
        connection_mefeatngraminput_output_.ForwardPropagate(mefeat_ngram_input_layer_, nce_output_layer_);
      }
      nce_output_layer_.ComputeActivations();
      logp = log(nce_output_layer_.activations(w));
      double sum = 0;
      size_t i;
      const size_t n = nce_output_layer_.nneurons();
      const vector<neuralnet::ActivationType> &ac = nce_output_layer_.activations();
      for (i = 0; i + 7 < n; ) {
        sum += ac[i];
        sum += ac[i + 1];
        sum += ac[i + 2];
        sum += ac[i + 3];
        sum += ac[i + 4];
        sum += ac[i + 5];
        sum += ac[i + 6];
        sum += ac[i + 7];
        i+= 8;
      }
      for (; i < n; ++i) {
        sum += ac[i];
      }
      logp -= log(sum);
    } else {
      nce_output_layer_.ResetInputForActivation(w);
      connection_hidden_output_.ForwardPropagateForOutput(hidden_layers_[0], nce_output_layer_, w);
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
      if (bias()) {
        connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
      }
      if (ngram_order_ > 1) {
        connection_mefeatngraminput_output_.ForwardPropagateForOutput(mefeat_ngram_input_layer_, nce_output_layer_, w);
      }
      logp = nce_output_layer_.activationinputs(w);
      // for numerical stability
      if (logp > 50) {
        logp = 50;
      }
      if (logp < -50) {
        logp = -50;
      }
    }
    // NCE normalizing constant is 1.
    // TODO: use more advanced NCE constants if unstable.
    return logp;
  }
}

} // namespace nnlm
