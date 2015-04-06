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
#include <utility>
#include <algorithm>
#include <limits>
#include "../neuralnet/futil.h"
#include "../neuralnet/neuralnet_numeric.h"
#include "maxent_fnnlm.h"

// <cstdio>
using std::size_t;
// <cstdlib>
using std::exit;
// <cmath>
using std::log;
using std::exp;
using std::floor;
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
// <utility>
using std::pair;
// <algorithm>
using std::sort;
using std::unique;
// <limits>
using std::numeric_limits;

namespace fnnlm {

void MaxEntFNeuralNetLM::ReadLMImpl(ifstream &ifs) {
  neuralnet::read_single(ifs, ngram_order_);
  cout << "ngram_order_: " << ngram_order_ << endl;
  neuralnet::read_single(ifs, hash_table_size_word_);
  cout << "hash_table_size_word_: " << hash_table_size_word_ << endl;
  neuralnet::read_single(ifs, hash_table_size_mixed_);
  cout << "hash_table_size_mixed_: " << hash_table_size_mixed_ << endl;
  neuralnet::read_single(ifs, hash_mode_);
  cout << "hash_mode_: " << hash_mode_ << endl;

  AllocateModel();

  connection_bias_output_.ReadConnection(ifs);

  if (ngram_order_ > 1) {
    connection_wordinput_output_.ReadConnection(ifs);

    if (use_factor_input()) {
      connection_mixedinput_output_.ReadConnection(ifs);
    }
  }

  if (globalbias()) {
    connection_globalbias_output_.ReadConnection(ifs);
  }

  if (use_factor_hidden()) {
    connection_bias_factorhidden_.ReadConnection(ifs);

    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_.ReadConnection(ifs);

      if (use_factor_input()) {
        connection_mixedinput_factorhidden_.ReadConnection(ifs);
      }
    }

    if (globalbias()) {
      connection_globalbias_factorhidden_.ReadConnection(ifs);
    }
  }
}

void MaxEntFNeuralNetLM::WriteLMImpl(ofstream &ofs) {
  neuralnet::write_single(ofs, ngram_order_);
  neuralnet::write_single(ofs, hash_table_size_word_);
  neuralnet::write_single(ofs, hash_table_size_mixed_);
  neuralnet::write_single(ofs, hash_mode_);

  connection_bias_output_.WriteConnection(ofs);

  if (ngram_order_ > 1) {
    connection_wordinput_output_.WriteConnection(ofs);

    if (use_factor_input()) {
      connection_mixedinput_output_.WriteConnection(ofs);
    }
  }

  if (globalbias()) {
    connection_globalbias_output_.WriteConnection(ofs);
  }

  if (use_factor_hidden()) {
    connection_bias_factorhidden_.WriteConnection(ofs);

    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_.WriteConnection(ofs);

      if (use_factor_input()) {
        connection_mixedinput_factorhidden_.WriteConnection(ofs);
      }
    }

    if (globalbias()) {
      connection_globalbias_factorhidden_.WriteConnection(ofs);
    }
  }
}

void MaxEntFNeuralNetLM::ExtractWordInputEmbeddingImpl(ofstream &ofs) {
}

void MaxEntFNeuralNetLM::ExtractWordOutputEmbeddingImpl(ofstream &ofs) {
}

void MaxEntFNeuralNetLM::PrintParamsImpl() {
  cout << "ngram_order_: " << ngram_order_ << endl;
  cout << "hash_table_size_word_: " << hash_table_size_word_ << endl;
  cout << "hash_table_size_mixed_: " << hash_table_size_mixed_ << endl;
  cout << "hash_mode_: " << hash_mode_ << endl;
}

void MaxEntFNeuralNetLM::CheckParamsImpl() {
  if (weight_factor_output() != 0) {
    cerr << "weight_factor_output invalid!" << endl;
    exit(EXIT_FAILURE);
  }

  if (ngram_order_ < 1) {
    cerr << "ngram order should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }
  if (ngram_order_ > 1) {
    if (hash_table_size_word_ < 1) {
      cerr << "hash table size (word) should be no less than 1 for ngram_order_ > 1!" << endl;
      exit(EXIT_FAILURE);
    }
    if (use_factor_input() && hash_table_size_mixed_ < 1) {
      cerr << "hash table size (mixed) should be no less than 1 for ngram_order_ > 1 and use_factor_input() = true!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  if (hash_mode_ != 0 && hash_mode_ != 1) {
    cerr << "hash mode can only be 0 or 1 in current implementation!" << endl;
    exit(EXIT_FAILURE);
  }
}

void MaxEntFNeuralNetLM::AllocateModel() {
  assert(ngram_order_  > 0);

  // ======================
  // Allocate layers.
  // ======================
  if (ngram_order_ > 1) {
    context_words_.set_capacity(ngram_order_ - 1);

    word_input_layer_.set_nneurons(hash_table_size_word_, true, false);
  
    if (use_factor_input()) {
      mixed_input_layer_.set_nneurons(hash_table_size_mixed_, true, false);
    }
  }

  if (nce()) {
    nce_output_layer_.set_nneurons(word_vocab().size(), false, true);
  } else {
    output_layer_.set_nneurons(word_vocab().size(), false, true);
  }

  bias_layer_.set_nneurons(1, true, false);
  bias_layer_.SetActivationsToValue(1.0);

  if (use_factor_hidden()) {
    factor_hidden_layer_.set_nneurons(factor_vocab().size(), false, true);
  }
  
  // ======================
  // Allocate connections.
  // ======================
  if (ngram_order_ > 1) {
    connection_wordinput_output_.set_dims(hash_table_size_word_, word_vocab().size());
    last_connection_wordinput_output_.set_dims(hash_table_size_word_, word_vocab().size());
    connection_wordinput_output_.set_adagrad(adagrad());
    last_connection_wordinput_output_.set_adagrad(adagrad());
    connection_wordinput_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_wordinput_output_.set_l2_regularization_param(l2_regularization_param());

    if (use_factor_input()) {
      connection_mixedinput_output_.set_dims(hash_table_size_mixed_, word_vocab().size());
      last_connection_mixedinput_output_.set_dims(hash_table_size_mixed_, word_vocab().size());
      connection_mixedinput_output_.set_adagrad(adagrad());
      last_connection_mixedinput_output_.set_adagrad(adagrad());
      connection_mixedinput_output_.set_l2_regularization_param(l2_regularization_param());
      last_connection_mixedinput_output_.set_l2_regularization_param(l2_regularization_param());
    }
  }

  // connection_bias_output_ uses storage_input_major_ = true is much faster.
  // (sort is time consuming).
  if (nce()) {
    connection_bias_output_.set_dims(false, 1, word_vocab().size());
    last_connection_bias_output_.set_dims(false, 1, word_vocab().size());
  } else {
    connection_bias_output_.set_dims(true, 1, word_vocab().size());
    last_connection_bias_output_.set_dims(true, 1, word_vocab().size());
  }
  connection_bias_output_.set_adagrad(adagrad());
  last_connection_bias_output_.set_adagrad(adagrad());
  connection_bias_output_.set_l2_regularization_param(l2_regularization_param());
  last_connection_bias_output_.set_l2_regularization_param(l2_regularization_param());

  if (globalbias()) {
    connection_globalbias_output_.set_dims(true, 1, word_vocab().size());
    last_connection_globalbias_output_.set_dims(true, 1, word_vocab().size());
    connection_globalbias_output_.set_adagrad(adagrad());
    last_connection_globalbias_output_.set_adagrad(adagrad());
    connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
    last_connection_globalbias_output_.set_l2_regularization_param(l2_regularization_param());
  }

  if (use_factor_hidden()) {
    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_.set_dims(hash_table_size_word_, factor_vocab().size());
      last_connection_wordinput_factorhidden_.set_dims(hash_table_size_word_, factor_vocab().size());
      connection_wordinput_factorhidden_.set_adagrad(adagrad());
      last_connection_wordinput_factorhidden_.set_adagrad(adagrad());
      connection_wordinput_factorhidden_.set_l2_regularization_param(l2_regularization_param());
      last_connection_wordinput_factorhidden_.set_l2_regularization_param(l2_regularization_param());

      if (use_factor_input()) {
        connection_mixedinput_factorhidden_.set_dims(hash_table_size_mixed_, factor_vocab().size());
        last_connection_mixedinput_factorhidden_.set_dims(hash_table_size_mixed_, factor_vocab().size());
        connection_mixedinput_factorhidden_.set_adagrad(adagrad());
        last_connection_mixedinput_factorhidden_.set_adagrad(adagrad());
        connection_mixedinput_factorhidden_.set_l2_regularization_param(l2_regularization_param());
        last_connection_mixedinput_factorhidden_.set_l2_regularization_param(l2_regularization_param());
      }
    }

    if (nce()) {
      connection_bias_factorhidden_.set_dims(false, 1, factor_vocab().size());
      last_connection_bias_factorhidden_.set_dims(false, 1, factor_vocab().size());
    } else {
      connection_bias_factorhidden_.set_dims(true, 1, factor_vocab().size());
      last_connection_bias_factorhidden_.set_dims(true, 1, factor_vocab().size());
    }
    connection_bias_factorhidden_.set_adagrad(adagrad());
    last_connection_bias_factorhidden_.set_adagrad(adagrad());
    connection_bias_factorhidden_.set_l2_regularization_param(l2_regularization_param());
    last_connection_bias_factorhidden_.set_l2_regularization_param(l2_regularization_param());

    if (globalbias()) {
      connection_globalbias_factorhidden_.set_dims(true, 1, factor_vocab().size());
      last_connection_globalbias_factorhidden_.set_dims(true, 1, factor_vocab().size());
      connection_globalbias_factorhidden_.set_adagrad(adagrad());
      last_connection_globalbias_factorhidden_.set_adagrad(adagrad());
      connection_globalbias_factorhidden_.set_l2_regularization_param(l2_regularization_param());
      last_connection_globalbias_factorhidden_.set_l2_regularization_param(l2_regularization_param());
    }
  }
}

void MaxEntFNeuralNetLM::InitializeNeuralNet() {
  // Reset activations.
  ResetActivations();

  // The connections are initialized as 0s.
  // Note: at least connection_input_output_ is initialized as all 0s.
}

void MaxEntFNeuralNetLM::ResetActivations() {
  // Also set context_words_.
  // TODO: should not insert </s>'s at the begining? should back-off?
  // "</s> </s> a" makes </s> counted twice!
  if (ngram_order_ > 1) {
    context_words_.set_offset(0);
    for (size_t i = 0; i < ngram_order_ - 1; i++) {
      context_words_[i].first = word_vocab().eos_idx();
      context_words_[i].second.clear();
      context_words_[i].second.push_back(factor_vocab().eos_idx());
    }
    word_input_layer_.SetActivationsToValue(0.0f);

    if (use_factor_input()) {
      mixed_input_layer_.SetActivationsToValue(0.0f);
    }
  }

  if (nce()) {
    nce_output_layer_.SetActivationsToValue(0.0f);
  } else {
    output_layer_.SetActivationsToValue(0.0f);
  }

  if (use_factor_hidden()) {
    factor_hidden_layer_.SetActivationsToValue(0.0f);
  }
}

void MaxEntFNeuralNetLM::CacheCurrentParams() {
  if (ngram_order_ > 1) {
    last_connection_wordinput_output_ = connection_wordinput_output_;
    if (use_factor_input()) {
      last_connection_mixedinput_output_ = connection_mixedinput_output_;
    }
  }
  last_connection_bias_output_ = connection_bias_output_;
  if (globalbias()) {
    last_connection_globalbias_output_ = connection_globalbias_output_;
  }

  if (use_factor_hidden()) {
    if (ngram_order_ > 1) {
      last_connection_wordinput_factorhidden_ = connection_wordinput_factorhidden_;
      if (use_factor_input()) {
        last_connection_mixedinput_factorhidden_ = connection_mixedinput_factorhidden_;
      }
    }
    last_connection_bias_factorhidden_ = connection_bias_factorhidden_;
    if (globalbias()) {
      last_connection_globalbias_factorhidden_ = connection_globalbias_factorhidden_;
    }
  }
}

void MaxEntFNeuralNetLM::RestoreLastParams() {
  if (ngram_order_ > 1) {
    connection_wordinput_output_ = last_connection_wordinput_output_;
    if (use_factor_input()) {
      connection_mixedinput_output_ = last_connection_mixedinput_output_;
    }
  }
  connection_bias_output_ = last_connection_bias_output_;
  if (globalbias()) {
    connection_globalbias_output_ = last_connection_globalbias_output_;
  }

  if (use_factor_hidden()) {
    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_ = last_connection_wordinput_factorhidden_;
      if (use_factor_input()) {
        connection_mixedinput_factorhidden_ = last_connection_mixedinput_factorhidden_;
      }
    }
    connection_bias_factorhidden_ = last_connection_bias_factorhidden_;
    if (globalbias()) {
      connection_globalbias_factorhidden_ = last_connection_globalbias_factorhidden_;
    }
  }
}

void MaxEntFNeuralNetLM::ForwardPropagate(size_t w, const vector<size_t> &fs) {
  if (ngram_order_ > 1) {
    context_words_.rotate(ngram_order_ - 2);
    context_words_[0].first = w;
    context_words_[0].second = fs;
    word_input_layer_.SetActivationsToValue(0);
    if (use_factor_input()) {
      mixed_input_layer_.SetActivationsToValue(0);
    }
    // Set n-gram max-ent features.
    if (hash_mode_ == 0) {
      const size_t n = word_vocab().size();
      size_t hidx_word = 5381;
      size_t hidx_mixed;
      vector<size_t> last_partial_hidx_mixed;
      vector<size_t> partial_hidx_mixed;
      for (size_t i = 0; i < ngram_order_ - 1; i++) {
        const size_t w = context_words_[i].first;
        const vector<size_t> &factors = context_words_[i].second; 

        vector<size_t> partial_hidx_factors;
        // update hidx for (words, factor) n-gram
        if (use_factor_input()) {
          for (vector<size_t>::const_iterator fit = factors.begin(); fit != factors.end(); ++fit) {
            hidx_mixed = hidx_word;
            neuralnet::hash_update0(hidx_mixed, *fit + n);
            partial_hidx_mixed.push_back(hidx_mixed);
            mixed_input_layer_.AccumulateActivation(hidx_mixed % hash_table_size_mixed_, 1);
          }
        }

        // pure word n-gram
        neuralnet::hash_update0(hidx_word, w);
        word_input_layer_.AccumulateActivation(hidx_word % hash_table_size_word_, 1);

        if (use_factor_input()) {
          for (vector<size_t>::const_iterator it = last_partial_hidx_mixed.begin(); it != last_partial_hidx_mixed.end(); ++it) {
            // update hidx for (mixed, word) n-gram
            hidx_mixed = *it;
            neuralnet::hash_update0(hidx_mixed, w);
            partial_hidx_mixed.push_back(hidx_mixed);
            mixed_input_layer_.AccumulateActivation(hidx_mixed % hash_table_size_mixed_, 1);

            // update hidx for (mixed, factor) n-gram
            for (vector<size_t>::const_iterator fit = partial_hidx_factors.begin(); fit != partial_hidx_factors.end(); ++fit) {
              hidx_mixed = *it;
              neuralnet::hash_update0(hidx_mixed, *fit + n);
              partial_hidx_mixed.push_back(hidx_mixed);
              mixed_input_layer_.AccumulateActivation(hidx_mixed % hash_table_size_mixed_, 1);
            }
          }

          last_partial_hidx_mixed = partial_hidx_mixed;
          partial_hidx_mixed.clear();
        }
      }
    } else if (hash_mode_ == 1) {
      const size_t n = word_vocab().size();
      for (size_t a = 1; a < ngram_order_; a++) {
        size_t hidx_word = neuralnet::HASH_OFFSET;
        vector<size_t> last_partial_hidx_mixed;
        vector<size_t> partial_hidx_mixed;
        for (size_t b = 1; b <= a; b++) {
          const size_t w = context_words_[b - 1].first;
          const vector<size_t> &factors = context_words_[b - 1].second; 
          
          size_t partial_hidx_word = 0;
          neuralnet::hash_update1(partial_hidx_word, a, b, w);

          vector<size_t> partial_hidx_factors;
          if (use_factor_input()) {
            for (vector<size_t>::const_iterator fit = factors.begin(); fit != factors.end(); ++fit) {
              size_t hidx = 0;
              neuralnet::hash_update1(hidx, a, b, *fit + n);
              partial_hidx_factors.push_back(hidx);
            }

            // update hidx for (words, factor) n-gram
            for (vector<size_t>::const_iterator fit = partial_hidx_factors.begin(); fit != partial_hidx_factors.end(); ++fit) {
              partial_hidx_mixed.push_back(hidx_word + *fit);
            }
          }

          // update hidx for pure word n-gram
          hidx_word += partial_hidx_word;
          if (use_factor_input()) {
            for (vector<size_t>::const_iterator it = last_partial_hidx_mixed.begin(); it != last_partial_hidx_mixed.end(); ++it) {
              // update hidx for (mixed, word) n-gram
              partial_hidx_mixed.push_back(*it + partial_hidx_word);
              // update hidx for (mixed, factor) n-gram
              for (vector<size_t>::const_iterator fit = partial_hidx_factors.begin(); fit != partial_hidx_factors.end(); ++fit) {
                partial_hidx_mixed.push_back(*it + *fit);
              }
            }

            last_partial_hidx_mixed = partial_hidx_mixed;
            partial_hidx_mixed.clear();
          }
        }

        word_input_layer_.set_activations(hidx_word % hash_table_size_word_, 1);

        if (use_factor_input()) {
          for (vector<size_t>::const_iterator it = last_partial_hidx_mixed.begin(); it != last_partial_hidx_mixed.end(); ++it) {
            mixed_input_layer_.AccumulateActivation(*it % hash_table_size_mixed_, 1);
          }
        }
      }
    } else {
      cerr << "Unknown hash mode!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (!nce()) {
    output_layer_.ResetInputForActivations();
    if (ngram_order_ > 1) {
      connection_wordinput_output_.ForwardPropagate(word_input_layer_, output_layer_);
      if (use_factor_input()) {
        connection_mixedinput_output_.ForwardPropagate(mixed_input_layer_, output_layer_);
      }
    }
    connection_bias_output_.ForwardPropagate(bias_layer_, output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.ForwardPropagate(bias_layer_, output_layer_);
    }

    if (use_factor_hidden()) {
      factor_hidden_layer_.ResetInputForActivations();
      if (ngram_order_ > 1) {
        connection_wordinput_factorhidden_.ForwardPropagate(word_input_layer_, factor_hidden_layer_);
        if (use_factor_input()) {
          connection_mixedinput_factorhidden_.ForwardPropagate(mixed_input_layer_, factor_hidden_layer_);
        }
      }
      connection_bias_factorhidden_.ForwardPropagate(bias_layer_, factor_hidden_layer_);
      if (globalbias()) {
        connection_globalbias_factorhidden_.ForwardPropagate(bias_layer_, factor_hidden_layer_);
      }
      // Note: skip the copy, but use activationinputs when activation is needed
      // factor_hidden_layer_.ComputeActivations();

      assert(output_layer_.nneurons() == word_vocab().size());
      for (size_t w = 0; w < word_vocab().size(); w++) {
        const vector<size_t> &factors = factors_for_word(w);
        for (vector<size_t>::const_iterator fit = factors.begin(); fit != factors.end(); ++fit) {
          output_layer_.AccumulateInputForActivation(w, factor_hidden_layer_.activationinputs(*fit));
        }
      }
    }
    output_layer_.ComputeActivations();
  }
}

void MaxEntFNeuralNetLM::BackPropagate(size_t w, const vector<size_t> &fs) {
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
        connection_wordinput_output_.ForwardPropagateForOutput(word_input_layer_, nce_output_layer_, it->first);
        if (use_factor_input()) {
          connection_mixedinput_output_.ForwardPropagateForOutput(mixed_input_layer_, nce_output_layer_, it->first);
        }
      }
      connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, it->first);
    }

    vector<size_t> nce_touched_factors;
    if (use_factor_hidden()) {
      // forward propagate hidden_layer_ -> factor_hidden_layer_.
      for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin(); 
           it != nce_sampled_words.end(); ++it) {
        const vector<size_t> &fs = factors_for_word(it->first);
        for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
          nce_touched_factors.push_back(*fit);
        }
      }
      sort(nce_touched_factors.begin(), nce_touched_factors.end());
      vector<size_t>::iterator last = unique(nce_touched_factors.begin(), nce_touched_factors.end());
      nce_touched_factors.erase(last, nce_touched_factors.end());
      factor_hidden_layer_.ResetInputForActivations();
      for (vector<size_t>::const_iterator fit = nce_touched_factors.begin(); fit != nce_touched_factors.end(); ++fit) {
        if (ngram_order_ > 1) {
          connection_wordinput_factorhidden_.ForwardPropagateForOutput(word_input_layer_, factor_hidden_layer_, *fit);
          if (use_factor_input()) {
            connection_mixedinput_factorhidden_.ForwardPropagateForOutput(mixed_input_layer_, factor_hidden_layer_, *fit);
          }
        }
        assert(globalbias());
        connection_globalbias_factorhidden_.ForwardPropagateForOutput(bias_layer_, factor_hidden_layer_, *fit);
        if (bias()) {
          connection_bias_factorhidden_.ForwardPropagateForOutput(bias_layer_, factor_hidden_layer_, *fit);
        }
      }
      // N.B. do not call ComputeActivations for factor_hidden_layer_.

      for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin(); 
           it != nce_sampled_words.end(); ++it) {
        const vector<size_t> &fs = factors_for_word(it->first);
        double val = 0;
        for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
          // N.B. rely on the fact that factor_hidden_layer_ is NeuralNetIdentityLayer.
          val += factor_hidden_layer_.activationinputs(*fit);
        }
        nce_output_layer_.AccumulateInputForActivation(it->first, val);
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
      log_kpnw = log_num_negative_samples() + log(word_noise_pdf(it->first));
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
    log_kpnw = log_num_negative_samples() + log(word_noise_pdf(w));
    log_den = log_kpnw;
    neuralnet::logadd(log_den, log_pw);
    rat = exp(log_kpnw - log_den);
    // assertion for numerical stability
    assert(rat >= 0 && rat <= 1.00001);
    nce_output_layer_.AccumulateError(w, rat);
    obj_val += log_pw - log_den;

    AccumulateNCEObjetiveValue(obj_val);

    if (use_factor_hidden()) {
      factor_hidden_layer_.SetErrorsToValue(0.0f);
      for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin();
           it != nce_sampled_words.end(); ++it) {
        neuralnet::ErrorType er = nce_output_layer_.errors(it->first);
        const vector<size_t> &fs = factors_for_word(it->first);
        for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
          factor_hidden_layer_.AccumulateError(*fit, er);
        }
      }
    }

    // no need to back-propagate error to input layer
    // no need to back-propagate error to bias layer

    for (unordered_map<size_t, int>::const_iterator it = nce_sampled_words.begin(); 
         it != nce_sampled_words.end(); ++it) {
      if (ngram_order_ > 1) {
        connection_wordinput_output_.AccumulateGradientsForOutput(word_input_layer_, nce_output_layer_, it->first);
        if (use_factor_input()) {
          connection_mixedinput_output_.AccumulateGradientsForOutput(mixed_input_layer_, nce_output_layer_, it->first);
        }
      }
      connection_bias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
      assert(globalbias());
      connection_globalbias_output_.AccumulateGradientsForOutput(bias_layer_, nce_output_layer_, it->first);
    }

    if (use_factor_hidden()) {
      for (vector<size_t>::const_iterator fit = nce_touched_factors.begin(); 
           fit != nce_touched_factors.end(); ++fit) {
        if (ngram_order_ > 1) {
          connection_wordinput_factorhidden_.AccumulateGradientsForOutput(word_input_layer_, factor_hidden_layer_, *fit);
          if (use_factor_input()) {
            connection_mixedinput_factorhidden_.AccumulateGradientsForOutput(mixed_input_layer_, factor_hidden_layer_, *fit);
          }
        }
        assert(globalbias());
        connection_globalbias_factorhidden_.AccumulateGradientsForOutput(bias_layer_, factor_hidden_layer_, *fit);
        if (bias()) {
          connection_bias_factorhidden_.AccumulateGradientsForOutput(bias_layer_, factor_hidden_layer_, *fit);
        }
      }
    }
  } else {
    const size_t n = word_vocab().size();
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

    if (use_factor_hidden()) {
      factor_hidden_layer_.SetErrorsToValue(0.0f);
      for (i = 0; i < n; i++) {
        neuralnet::ErrorType er = output_layer_.errors(i);
        const vector<size_t> &fs = factors_for_word(i);
        for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
          factor_hidden_layer_.AccumulateError(*fit, er);
        }
      }
    }

    // no need to back-propagate error to input layer
    // no need to back-propagate error to bias layer
    
    if (ngram_order_ > 1) {
      connection_wordinput_output_.AccumulateGradients(word_input_layer_, output_layer_);
      if (use_factor_input()) {
        connection_mixedinput_output_.AccumulateGradients(mixed_input_layer_, output_layer_);
      }
    }
    connection_bias_output_.AccumulateGradients(bias_layer_, output_layer_);
    if (globalbias()) {
      connection_globalbias_output_.AccumulateGradients(bias_layer_, output_layer_);
    }

    if (use_factor_hidden()) {
      if (ngram_order_ > 1) {
        connection_wordinput_factorhidden_.AccumulateGradients(word_input_layer_, factor_hidden_layer_);
        if (use_factor_input()) {
          connection_mixedinput_factorhidden_.AccumulateGradients(mixed_input_layer_, factor_hidden_layer_);
        }
      }
      connection_bias_factorhidden_.AccumulateGradients(bias_layer_, factor_hidden_layer_);
      if (globalbias()) {
        connection_globalbias_factorhidden_.AccumulateGradients(bias_layer_, factor_hidden_layer_);
      }
    }
  }
}

void MaxEntFNeuralNetLM::FastUpdateWeightsMajor(float learning_rate) {
  if (ngram_order_ > 1) {
    connection_wordinput_output_.FastUpdateWeightsMajor(learning_rate);
    if (use_factor_input()) {
      connection_mixedinput_output_.FastUpdateWeightsMajor(learning_rate);
    }
  }
  connection_bias_output_.FastUpdateWeightsMajor(learning_rate);
  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMajor(learning_rate);
  }

  if (use_factor_hidden()) {
    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_.FastUpdateWeightsMajor(learning_rate);
      if (use_factor_input()) {
        connection_mixedinput_factorhidden_.FastUpdateWeightsMajor(learning_rate);
      }
    }
    connection_bias_factorhidden_.FastUpdateWeightsMajor(learning_rate);
    if (globalbias()) {
      connection_globalbias_factorhidden_.FastUpdateWeightsMajor(learning_rate);
    }
  }
}

void MaxEntFNeuralNetLM::FastUpdateWeightsMinor() {
  if (ngram_order_ > 1) {
    connection_wordinput_output_.FastUpdateWeightsMinor();
    if (use_factor_input()) {
      connection_mixedinput_output_.FastUpdateWeightsMinor();
    }
  }
  connection_bias_output_.FastUpdateWeightsMinor();
  if (globalbias()) {
    connection_globalbias_output_.FastUpdateWeightsMinor();
  }

  if (use_factor_hidden()) {
    if (ngram_order_ > 1) {
      connection_wordinput_factorhidden_.FastUpdateWeightsMinor();
      if (use_factor_input()) {
        connection_mixedinput_factorhidden_.FastUpdateWeightsMinor();
      }
    }
    connection_bias_factorhidden_.FastUpdateWeightsMinor();
    if (globalbias()) {
      connection_globalbias_factorhidden_.FastUpdateWeightsMinor();
    }
  }
}

double MaxEntFNeuralNetLM::GetLogProb(size_t w, bool nce_exact) {
  if (!nce()) {
    // Ignore nce_exact.
    return log(output_layer_.activations(w));
  } else {
    double logp;
    if (nce_exact) {
      size_t i;
      const size_t n = nce_output_layer_.nneurons();
      nce_output_layer_.ResetInputForActivations();
      if (ngram_order_ > 1) {
        connection_wordinput_output_.ForwardPropagate(word_input_layer_, nce_output_layer_);
        if (use_factor_input()) {
          connection_mixedinput_output_.ForwardPropagate(mixed_input_layer_, nce_output_layer_);
        }
      }
      connection_bias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagate(bias_layer_, nce_output_layer_);
      if (use_factor_hidden()) {
        factor_hidden_layer_.ResetInputForActivations();
        if (ngram_order_ > 1) {
          connection_wordinput_factorhidden_.ForwardPropagate(word_input_layer_, factor_hidden_layer_);
          if (use_factor_input()) {
            connection_mixedinput_factorhidden_.ForwardPropagate(mixed_input_layer_, factor_hidden_layer_);
          }
        }
        connection_bias_factorhidden_.ForwardPropagate(bias_layer_, factor_hidden_layer_);
        assert(globalbias());
        connection_globalbias_factorhidden_.ForwardPropagate(bias_layer_, factor_hidden_layer_);

        for (i = 0; i < n; i++) {
          const vector<size_t> &fs = factors_for_word(i);
          double val = 0;
          for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
            val += factor_hidden_layer_.activationinputs(*fit);
          }
          nce_output_layer_.AccumulateInputForActivation(i, val);
        }
      }
      nce_output_layer_.ComputeActivations();
      logp = log(nce_output_layer_.activations(w));
      double sum = 0;
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
        connection_wordinput_output_.ForwardPropagateForOutput(word_input_layer_, nce_output_layer_, w);
        if (use_factor_input()) {
          connection_mixedinput_output_.ForwardPropagateForOutput(mixed_input_layer_, nce_output_layer_, w);
        }
      }
      connection_bias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
      assert(globalbias());
      connection_globalbias_output_.ForwardPropagateForOutput(bias_layer_, nce_output_layer_, w);
      if (use_factor_hidden()) {
        const vector<size_t> &fs = factors_for_word(w);
        factor_hidden_layer_.ResetInputForActivations();
        for (vector<size_t>::const_iterator fit = fs.begin(); fit != fs.end(); ++fit) {
          if (ngram_order_ > 1) {
            connection_wordinput_factorhidden_.ForwardPropagateForOutput(word_input_layer_, factor_hidden_layer_, *fit);
            if (use_factor_input()) {
              connection_mixedinput_factorhidden_.ForwardPropagateForOutput(mixed_input_layer_, factor_hidden_layer_, *fit);
            }
          }
          connection_bias_factorhidden_.ForwardPropagateForOutput(bias_layer_, factor_hidden_layer_, *fit);
          assert(globalbias());
          connection_globalbias_factorhidden_.ForwardPropagateForOutput(bias_layer_, factor_hidden_layer_, *fit);
          nce_output_layer_.AccumulateInputForActivation(w, factor_hidden_layer_.activationinputs(*fit));
        }
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

} // namespace fnnlm
