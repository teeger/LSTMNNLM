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
#include "maxent_nnlm.h"

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

void MaxEntNeuralNetLM::ReadLMImpl(ifstream &ifs) {
  neuralnet::read_single(ifs, ngram_order_);
  cout << "ngram_order_: " << ngram_order_ << endl;
  neuralnet::read_single(ifs, hash_table_size_);
  cout << "hash_table_size_: " << hash_table_size_ << endl;
  neuralnet::read_single(ifs, hash_mode_);
  cout << "hash_mode_: " << hash_mode_ << endl;

  AllocateModel();

  if (ngram_order_ > 1) {
    connection_input_output_.ReadConnection(ifs);
  }
  connection_bias_output_.ReadConnection(ifs);

  if (globalbias()) {
    connection_globalbias_output_.ReadConnection(ifs);
  }
}

void MaxEntNeuralNetLM::WriteLMImpl(ofstream &ofs) {
  neuralnet::write_single(ofs, ngram_order_);
  neuralnet::write_single(ofs, hash_table_size_);
  neuralnet::write_single(ofs, hash_mode_);

  if (ngram_order_ > 1) {
    connection_input_output_.WriteConnection(ofs);
  }
  connection_bias_output_.WriteConnection(ofs);

  if (globalbias()) {
    connection_globalbias_output_.WriteConnection(ofs);
  }
}

void MaxEntNeuralNetLM::ExtractWordInputEmbeddingImpl(ofstream &ofs) {
}

void MaxEntNeuralNetLM::ExtractWordOutputEmbeddingImpl(ofstream &ofs) {
}

void MaxEntNeuralNetLM::PrintParamsImpl() {
  cout << "ngram_order_: " << ngram_order_ << endl;
  cout << "hash_table_size_: " << hash_table_size_ << endl;
  cout << "hash_mode_: " << hash_mode_ << endl;
}

void MaxEntNeuralNetLM::CheckParamsImpl() {
  if (ngram_order_ < 1) {
    cerr << "ngram order should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }
  if (ngram_order_ > 1 && hash_table_size_ < 1) {
    cerr << "hash table size should be no less than 1 for ngram_order_ > 1!" << endl;
    exit(EXIT_FAILURE);
  }
  if (hash_mode_ != 0 && hash_mode_ != 1) {
    cerr << "hash mode can only be 0 or 1 in current implementation!" << endl;
    exit(EXIT_FAILURE);
  }
}

void MaxEntNeuralNetLM::AllocateModel() {
  assert(ngram_order_  > 0);

  // Allocate context_words_
  if (ngram_order_ > 1) {
    context_words_.set_capacity(ngram_order_ - 1);
    input_layer_.set_nneurons(hash_table_size_, true, false);
  }


  if (nce()) {
    nce_output_layer_.set_nneurons(vocab().size(), false, true);
  } else {
    output_layer_.set_nneurons(vocab().size(), false, true);
  }

  bias_layer_.set_nneurons(1, true, false);
  bias_layer_.SetActivationsToValue(1.0);
  
  if (ngram_order_ > 1) {
    connection_input_output_.set_dims(hash_table_size_, vocab().size());
    last_connection_input_output_.set_dims(hash_table_size_, vocab().size());
    connection_input_output_.set_adagrad(adagrad());
    last_connection_input_output_.set_adagrad(adagrad());
    connection_input_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_input_output_.set_l2_regularization_param(l2_regularization_param());
  }

  // connection_bias_output_ uses storage_input_major_ = true is much faster.
  // (sort is time consuming).
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

  if (globalbias()) {
    connection_globalbias_output_.set_dims(true, 1, vocab().size());
    last_connection_globalbias_output_.set_dims(true, 1, vocab().size());
    connection_globalbias_output_.set_adagrad(adagrad());
    last_connection_globalbias_output_.set_adagrad(adagrad());
    connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
  }
}

void MaxEntNeuralNetLM::InitializeNeuralNet() {
  // Reset activations.
  ResetActivations();

  // The connections are initialized as 0s.
  // Note: at least connection_input_output_ is initialized as all 0s.
}

void MaxEntNeuralNetLM::ResetActivations() {
  // Also set context_words_.
  // TODO: should not insert </s>'s at the begining? should back-off?
  // "</s> </s> a" makes </s> counted twice!
  if (ngram_order_ > 1) {
    context_words_.set_offset(0);
    for (size_t i = 0; i < ngram_order_ - 1; i++) {
      context_words_[i] = vocab().eos_idx();
    }
    input_layer_.SetActivationsToValue(0.0f);
  }

  if (nce()) {
    nce_output_layer_.SetActivationsToValue(0.0f);
  } else {
    output_layer_.SetActivationsToValue(0.0f);
  }
}

void MaxEntNeuralNetLM::CacheCurrentParams() {
  if (ngram_order_ > 1) {
    last_connection_input_output_ = connection_input_output_;
  }
  last_connection_bias_output_ = connection_bias_output_;
  if (globalbias()) {
    last_connection_globalbias_output_ = connection_globalbias_output_;
  }
}

void MaxEntNeuralNetLM::RestoreLastParams() {
  if (ngram_order_ > 1) {
    connection_input_output_ = last_connection_input_output_;
  }
  connection_bias_output_ = last_connection_bias_output_;
  if (globalbias()) {
    connection_globalbias_output_ = last_connection_globalbias_output_;
  }
}

void MaxEntNeuralNetLM::ForwardPropagate(size_t w) {
  if (ngram_order_ > 1) {
    context_words_.rotate(ngram_order_ - 2);
    context_words_[0] = w;
    input_layer_.SetActivationsToValue(0);
    // Set n-gram max-ent features.
    if (hash_mode_ == 0) {
      size_t hv = 5381;
      for (size_t i = 0; i < ngram_order_ - 1; i++) {
        neuralnet::hash_update0(hv, context_words_[i]);
        size_t hidx = hv % hash_table_size_;
        input_layer_.AccumulateActivation(hidx, 1);
      }
    } else if (hash_mode_ == 1) {
      for (size_t a = 1; a < ngram_order_; a++) {
        size_t hidx = neuralnet::HASH_OFFSET;
        for (size_t b = 1; b <= a; b++) {
          neuralnet::hash_update1(hidx, a, b, context_words_[b - 1]);
        }
        hidx = hidx % hash_table_size_;
        input_layer_.AccumulateActivation(hidx, 1);
      }
    } else {
      cerr << "Unknown hash mode!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (!nce()) {
    output_layer_.ResetInputForActivations();
    if (ngram_order_ > 1) {
      connection_input_output_.ForwardPropagate(input_layer_, output_layer_);
    }
    connection_bias_output_.ForwardPropagate(bias_layer_, output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.ForwardPropagate(bias_layer_, output_layer_);
    }
    output_layer_.ComputeActivations();
  }
}

void MaxEntNeuralNetLM::BackPropagate(size_t w) {
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

    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
      it != nce_sampled_words.end(); ++it) {
      nce_output_layer_.ResetInputForActivation(it->first);
      if (ngram_order_ > 1) {
        connection_input_output_.ForwardPropagateForOutput(input_layer_, nce_output_layer_, it->first);
      }
      connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
    }

    // N.B.: directly set errors to avoid SetErrorsToValue.
    // But should be very careful about postive sample.
    double log_pw, log_kpnw, log_den, rat;
    double obj_val = 0;
    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
         it != nce_sampled_words.end(); ++it) {
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

    // no need to back-propagate error to input layer
    // no need to back-propagate error to bias layer

    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin(); 
         it != nce_sampled_words.end(); ++it) {
      if (ngram_order_ > 1) {
        connection_input_output_.AccumulateGradientsForOutput(input_layer_, nce_output_layer_, it->first);
      }
      connection_bias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
      assert(globalbias());
      connection_globalbias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
    }
  } else {
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

    // no need to back-propagate error to input layer
    // no need to back-propagate error to bias layer
    
    if (ngram_order_ > 1) {
      connection_input_output_.AccumulateGradients(input_layer_, output_layer_);
    }
    connection_bias_output_.AccumulateGradients(bias_layer_, output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.AccumulateGradients(bias_layer_, output_layer_);
    }
  }
}

void MaxEntNeuralNetLM::FastUpdateWeightsMajor(float learning_rate) {
  if (ngram_order_ > 1) {
    connection_input_output_.FastUpdateWeightsMajor(learning_rate);
  }
  connection_bias_output_.FastUpdateWeightsMajor(learning_rate);
  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMajor(learning_rate);
  }
}

void MaxEntNeuralNetLM::FastUpdateWeightsMinor() {
  if (ngram_order_ > 1) {
    connection_input_output_.FastUpdateWeightsMinor();
  }
  connection_bias_output_.FastUpdateWeightsMinor();
  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMinor();
  }
}

double MaxEntNeuralNetLM::GetLogProb(size_t w, bool nce_exact) {
  if (!nce()) {
    // Ignore nce_exact.
    return log(output_layer_.activations(w));
  } else {
    double logp;
    if (nce_exact) {
      nce_output_layer_.ResetInputForActivations();
      if (ngram_order_ > 1) {
        connection_input_output_.ForwardPropagate(input_layer_, nce_output_layer_);
      }
      connection_bias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
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
      if (ngram_order_ > 1) {
        connection_input_output_.ForwardPropagateForOutput(input_layer_, nce_output_layer_, w);
      }
      connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
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
