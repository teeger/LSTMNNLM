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
#include <utility>
#include <limits>
#include <boost/tokenizer.hpp>
#include "../neuralnet/futil.h"
#include "fnnlm_base.h"

// <cstdio>
using std::size_t;
// <cstdlib>
using std::exit;
// <cmath>
using std::log;
using std::exp;
using std::sqrt;
using std::abs;
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
using std::to_string;
// <vector>
using std::vector;
// <utility>
using std::pair;
// <limits>
using std::numeric_limits;
// <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;


namespace fnnlm {

void FNeuralNetLMBase::TrainLM(const string &validationfile,
                               const string &outbase,
                               bool nce_ppl) {
  // =============
  // Prepare for the training
  // Equivalent to ReadLM
  word_vocab_.ReadVocabFromTxt(word_vocab_filename_);
  if (word_vocab_.empty()) {
    cerr << "empty word vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  factor_vocab_.ReadVocabFromTxt(factor_vocab_filename_);
  if (factor_vocab_.empty()) {
    cerr << "empty factor vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  ReadDecompFromTxt(decomp_filename_);

  PrintParams();
  CheckParams();
  AllocateModel();
  InitializeNeuralNet();
  // ==== END ====

  // Read the data
  FNNLMDataReader train_data(train_filenames_, &word_vocab_, &factor_vocab_,
                             shuffle_datafiles_, shuffle_sentences_);
  vector<string> validation_filenames = { validationfile };
  FNNLMDataReader validation_data(validation_filenames, &word_vocab_, &factor_vocab_, false, false);

  // Set NCE sampling.
  if (nce_) {
    // TODO: flatten noise_distribution_?
    vector<int> word_count(word_vocab_.size(), 0);
    int num_word_tokens = 0;
    const size_t eos_widx = word_vocab().eos_idx();
    vector<int> factor_count(factor_vocab_.size(), 0);
    int num_factor_tokens = 0;
    const size_t eos_fidx = factor_vocab().eos_idx();

    vector<pair<size_t, vector<size_t>>> sentence;

    train_data.StartEpoch();
    while(train_data.GetSentence(sentence)) {
      for (vector<pair<size_t, vector<size_t>>>::const_iterator it = sentence.begin(); it != sentence.end(); ++it) {
        word_count[it->first]++;
        num_word_tokens++;
        if (weight_factor_output_ > 0) {
          for (size_t p = 0; p < it->second.size(); p++) {
            factor_count[it->second[p]]++;
            num_factor_tokens++;
          }
        }
      }
      word_count[eos_widx]++;
      num_word_tokens++;
      if (weight_factor_output_ > 0) {
        factor_count[eos_fidx]++;
        num_factor_tokens++;
      }
    }

    word_noise_distribution_ = Distribution(word_count.begin(), word_count.end());
    word_noise_pdf_ = word_noise_distribution_.param().probabilities();
    if (weight_factor_output_ > 0) {
      factor_noise_distribution_ = Distribution(factor_count.begin(), factor_count.end());
      factor_noise_pdf_ = factor_noise_distribution_.param().probabilities();
    }
    NCECheckSampling();
    log_num_negative_samples_ = log(num_negative_samples_);
  }

  BatchSGDTrain(train_data, validation_data, outbase, nce_ppl);

  cout << "================================================================================" << endl;
  cout << "Log-likelihood (base e) on validation is: " \
      << EvalLM(validation_data, false) << endl;
}

void FNeuralNetLMBase::EvalLM(const string &infile, bool nce_ppl) {
  CheckParams();
  vector<string> filenames = { infile };
  FNNLMDataReader data(filenames, &word_vocab_, &factor_vocab_, false, false);
  cout << "log-likelihood (base e) on " << infile << " is: " << EvalLM(data, nce_ppl) << endl;
}

void FNeuralNetLMBase::ReadLM(const string &infile) {
  cout << "============================" << endl;
  cout << "reading " << infile << endl;
  ifstream ifs;
  ifs.open(infile, ios::binary | ios::in);
  if (ifs.fail()) {
    cout << "unable to open " << infile << endl;
    exit(EXIT_FAILURE);
  }

  // TODO: read other parameters?

  neuralnet::read_single(ifs, l2_regularization_param_);
  cout << "l2_regularization_param_: " << l2_regularization_param_ << endl;
  neuralnet::read_single(ifs, adagrad_);
  cout << "adagrad_: " << adagrad_ << endl;

  neuralnet::read_single(ifs, nce_);
  cout << "nce_: " << nce_ << endl;
  neuralnet::read_single(ifs, num_negative_samples_);
  cout << "num_negative_samples_: " << num_negative_samples_ << endl;

  neuralnet::read_single(ifs, independent_);
  cout << "independent_: " << independent_ << endl;
  neuralnet::read_single(ifs, globalbias_);
  cout << "globalbias_: " << globalbias_ << endl;
  neuralnet::read_single(ifs, bias_);
  cout << "bias_: " << bias_ << endl;
  neuralnet::read_single(ifs, errorinput_cutoff_);
  cout << "errorinput_cutoff_: " << errorinput_cutoff_ << endl;

  neuralnet::read_single(ifs, use_factor_input_);
  cout << "use_factor_input_: " << use_factor_input_ << endl;
  neuralnet::read_single(ifs, use_factor_hidden_);
  cout << "use_factor_hidden_: " << use_factor_hidden_ << endl;
  neuralnet::read_single(ifs, weight_factor_output_);
  cout << "weight_factor_output_: " << weight_factor_output_ << endl;

  word_vocab_.ReadVocab(ifs);
  if (word_vocab_.empty()) {
    cerr << "empty word vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  factor_vocab_.ReadVocab(ifs);
  if (factor_vocab_.empty()) {
    cerr << "empty word vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  neuralnet::read_2d_vector(ifs, factors_for_word_);

  ReadLMImpl(ifs);

  // try one more read before check eof
  ifs.get(); 
  if (!ifs.eof()) {
    cout << "Error reading model: should be at the end of the file" << endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
}

void FNeuralNetLMBase::WriteLM(const string &outbase) {
  string modelname(outbase);
  modelname += ".model";
  cout << "writing " << modelname << endl;
  ofstream ofs;
  ofs.open(modelname, ios::binary | ios::out);
  if (ofs.fail()) {
    cout << "unable to open " << modelname << endl;
    exit(EXIT_FAILURE);
  }

  // TODO: write other parameters?

  neuralnet::write_single(ofs, l2_regularization_param_);
  neuralnet::write_single(ofs, adagrad_);

  neuralnet::write_single(ofs, nce_);
  neuralnet::write_single(ofs, num_negative_samples_);

  neuralnet::write_single(ofs, independent_);
  neuralnet::write_single(ofs, globalbias_);
  neuralnet::write_single(ofs, bias_);
  neuralnet::write_single(ofs, errorinput_cutoff_);

  neuralnet::write_single(ofs, use_factor_input_);
  neuralnet::write_single(ofs, use_factor_hidden_);
  neuralnet::write_single(ofs, weight_factor_output_);

  word_vocab_.WriteVocab(ofs);
  factor_vocab_.WriteVocab(ofs);
  neuralnet::write_2d_vector(ofs, factors_for_word_);

  WriteLMImpl(ofs);

  ofs.close();
}

void FNeuralNetLMBase::ExtractWordInputEmbedding(const string &filename) {
  CheckParams();

  cout << "writing " << filename << endl;
  ofstream ofs;
  ofs.open(filename, ios::out);
  if (ofs.fail()) {
    cout << "unable to open " << filename << endl;
    exit(EXIT_FAILURE);
  }

  ExtractWordInputEmbeddingImpl(ofs);

  ofs.close();
}

void FNeuralNetLMBase::ExtractWordOutputEmbedding(const string &filename) {
  CheckParams();

  cout << "writing " << filename << endl;
  ofstream ofs;
  ofs.open(filename, ios::out);
  if (ofs.fail()) {
    cout << "unable to open " << filename << endl;
    exit(EXIT_FAILURE);
  }

  ExtractWordOutputEmbeddingImpl(ofs);

  ofs.close();
}

void FNeuralNetLMBase::ReadDecompFromTxt(const string &decomptxt) {
  if (decomptxt == "") {
    cerr << "Warning: no decomposition file is set!" << endl;
    factors_for_word_.clear();
  } else {
    ifstream ifs;
    ifs.open(decomptxt, ios::in);
    if (ifs.fail()) {
      cerr << "Unable to open " << decomptxt << endl;
      exit(EXIT_FAILURE);
    }

    factors_for_word_.resize(word_vocab_.size());

    string line;
    char_separator<char> space_separator(" ");
    char_separator<char> tab_separator("\t");
    while (getline(ifs, line)) {
      tokenizer<char_separator<char>> word_factors(line, tab_separator);
      tokenizer<char_separator<char>>::const_iterator it = word_factors.begin();
      size_t w = word_vocab_.idx4type(*it);
      vector<size_t> &fs = factors_for_word_[w];
      if (!fs.empty()) {
        cerr << "double defined decomposition for word " << *it << endl;
        exit(EXIT_FAILURE);
      }

      if (++it == word_factors.end()) {
        cerr << "wrong decomposition format (no tab)!" << endl;
        exit(EXIT_FAILURE);
      }
      tokenizer<char_separator<char>> factors(*it, tab_separator);
      if(++it != word_factors.end()) {
        cerr << "wrong decomposition format (more than one tab)!" << endl;
        exit(EXIT_FAILURE);
      }

      for (it = factors.begin(); it != factors.end(); ++it) {
        fs.push_back(factor_vocab_.idx4type(*it));
      }
      assert(!fs.empty());
    }

    for (size_t w = 0; w < word_vocab_.size(); w++) {
      if (factors_for_word_[w].empty()) {
        cerr << "no decomposition defined for word: " << word_vocab_.type4idx(w) << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
}

void FNeuralNetLMBase::PrintParams() {
  cout << "unk_: " << unk_ << endl;
  cout << "word_vocab_filename_: " << word_vocab_filename_ << endl;
  cout << "word_vocab_.size_: " << word_vocab_.size() << endl;
  cout << "word_vocab_.eos_idx_: " << word_vocab_.eos_idx() << endl;
  cout << "word_vocab_.unk_idx_: " << word_vocab_.unk_idx() << endl;
  cout << "factor_vocab_filename_: " << factor_vocab_filename_ << endl;
  cout << "factor_vocab_.size_: " << factor_vocab_.size() << endl;
  cout << "factor_vocab_.eos_idx_: " << factor_vocab_.eos_idx() << endl;
  cout << "factor_vocab_.unk_idx_: " << factor_vocab_.unk_idx() << endl;
  cout << "decomp_filename_: " << decomp_filename_ << endl;
  cout << "trainfile_names_:";
  for (vector<string>::const_iterator it = train_filenames_.begin();
       it != train_filenames_.end(); ++it) {
    cout << " " << *it;
  }
  cout << endl;
  cout << "shuffle_datafiles_: " << shuffle_datafiles_ << endl;
  cout << "shuffle_sentences_: " << shuffle_sentences_ << endl;
  cout << "algopts_.init_learning_rate_: " << algopts_.init_learning_rate_ << endl;
  cout << "algopts_.batch_size_: " << algopts_.batch_size_ << endl;
  cout << "algopts_.min_improvement_: " << algopts_.min_improvement_ << endl;

  cout << "l2_regularization_param_: " << l2_regularization_param_ << endl;
  cout << "adagrad_: " << adagrad_ << endl;

  cout << "nce_: " << nce_ << endl;
  cout << "num_negative_samples_: " << num_negative_samples_ << endl;

  cout << "independent_: " << independent_ << endl;
  cout << "globalbias_: " << globalbias_ << endl;
  cout << "bias_: " << bias_ << endl;
  cout << "errorinput_cutoff_: " << errorinput_cutoff_ << endl;

  cout << "use_factor_input_: " << use_factor_input_ << endl;
  cout << "use_factor_hidden_: " << use_factor_hidden_ << endl;
  cout << "weight_factor_output_: " << weight_factor_output_ << endl;

  PrintParamsImpl();
}

void FNeuralNetLMBase::CheckParams() {
  if (word_vocab_.empty()) {
    cerr << "word vocabulary is empty!" << endl;
    exit(EXIT_FAILURE);
  }
  if (factor_vocab_.empty()) {
    cerr << "factor vocabulary is empty!" << endl;
    exit(EXIT_FAILURE);
  }
  if (algopts_.init_learning_rate_ <= 0) {
    cerr << "initial learning rate should be greater than 0!" << endl;
    exit(EXIT_FAILURE);
  }
  if (algopts_.batch_size_ <= 0) {
    cerr << "batch size should be greater than 0!" << endl;
    exit(EXIT_FAILURE);
  }
  if (algopts_.min_improvement_ < 1) {
    cerr << "min-improvement should be no less than 1!" << endl;
    exit(EXIT_FAILURE);
  }

  if (l2_regularization_param_ < 0) {
    cerr << "l2 regularization parameter should be no less than 0!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!nce_ && num_negative_samples_ > 0) {
    cerr << "num_negative_samples_ should be 0 when nce_ is false!" << endl;
    exit(EXIT_FAILURE);
  }

  if (nce_ && !globalbias_) {
    cerr << "globalbias_ is required for NCE, otherwise unstable!" << endl;
    exit(EXIT_FAILURE);
  }
  if (bias_ && !globalbias_) {
    cerr << "globalbias_ is recommended when bias_ is enabled!" << endl;
    exit(EXIT_FAILURE);
  }
  if (errorinput_cutoff_ < 0) {
    cerr << "errorinput_cutoff should be non-negative!" << endl;
    exit(EXIT_FAILURE);
  }

  if (use_factor_hidden_ && factors_for_word_.size() == 0) {
    cerr << "factors_for_word_ should not be empty when use_factor_hidden_ is true!" << endl;
    exit(EXIT_FAILURE);
  }
  if (weight_factor_output_ > 0 && factor_vocab_.size() == 0) {
    cerr << "weight_factor_output_ > 0 but factor_vocab_ is empty!" << endl;
    exit(EXIT_FAILURE);
  }

  CheckParamsImpl();
}

double FNeuralNetLMBase::EvalLM(FNNLMDataReader &data, bool nce_ppl) {
  CheckParams();

  if (nce_ppl && !nce_) {
    cerr << "nce_ppl == true but nce_ is false!" << endl;
    exit(EXIT_FAILURE);
  }

  const size_t eos_widx = word_vocab_.eos_idx();
  const size_t unk_widx = word_vocab_.unk_idx();
  const vector<size_t> eos_fidx = { factor_vocab_.eos_idx() };

  vector<pair<size_t, vector<size_t>>> sentence;
  double total_logp = 0.0;
  size_t sents_processed = 0;
  size_t ivcount = 0;
  size_t oovcount = 0;

  data.StartEpoch();

  ResetActivations();

  while (data.GetSentence(sentence)) {
    assert(!sentence.empty());

    if (independent_) {
      ResetActivations();
    }
    double curr_logp = 0.0;
    ForwardPropagate(eos_widx, eos_fidx);
    for (vector<pair<size_t, vector<size_t>>>::const_iterator it = sentence.begin(); it != sentence.end(); ++it) {
      if (!unk_ && it->first == unk_widx) {
        oovcount++;
      } else {
        curr_logp += GetLogProb(it->first, !nce_ppl);
        ivcount++;
      }
      ForwardPropagate(it->first, it->second);
    }
    curr_logp += GetLogProb(eos_widx, !nce_ppl);
    ivcount++;

    total_logp += curr_logp;
    sents_processed++;
    if ((sents_processed % 200) == 0) {
      cout << "." << flush;
    }

    if (debug_ > 1) {
      if (nce_ppl) {
        cerr << "unnormalized log-likelihood (base e) on " << sents_processed << "-th sentence is: " << curr_logp << endl;
      } else {
        cerr << "log-likelihood (base e) on " << sents_processed << "-th sentence is: " << curr_logp << endl;
      }
    }
  }

  if (ivcount == 0) {
    cerr << "zero IV words!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "\nnumber of IV words  (including </s>): " << ivcount << endl;
  cout << "number of OOV words: " << oovcount << endl;
  cout << "entropy (base 2): " << -total_logp / log(2) / ivcount << endl;
  if (nce_ppl) {
    cout << "unnormalied model perplexity: " << exp(-total_logp / ivcount) << endl;
  } else {
    cout << "model perplexity: " << exp(-total_logp / ivcount) << endl;
  }

  return total_logp;
}

void FNeuralNetLMBase::BatchSGDTrain(FNNLMDataReader &train_data, FNNLMDataReader &validation_data,
                                     const string &outbase, bool nce_ppl) {
  const size_t eos_widx = word_vocab_.eos_idx();
  const size_t unk_widx = word_vocab_.unk_idx();
  const vector<size_t> eos_fidx = { factor_vocab_.eos_idx() };

  vector<pair<size_t, vector<size_t>>> sentence;

  double last_logp = -numeric_limits<double>::max();
  double curr_logp = -numeric_limits<double>::max();
  bool halve_alpha = false;
  // set the current learning rate.
  float curr_learning_rate = algopts_.init_learning_rate_;

  size_t sents_processed = 0;
  int iteration = 0;

  clock_t start_time = clock();
  clock_t end_time = start_time;
  while (true) {
    cout << "******************************* ITERATION " << iteration++ << " *******************************" << endl;

    train_data.StartEpoch();

    ResetActivations();

    cout << "learning_rate = " << curr_learning_rate << endl;

    int bpos = 0;
    double logp = 0.0;
    nce_obj_ = 0;
    size_t ivcount = 0;
    size_t oovcount = 0;
    // NOTE: the vector "sentence" does not include </s> at the end!
    while (train_data.GetSentence(sentence)) {
      assert(!sentence.empty());

      if (independent_) {
        ResetActivations();
      }
      ForwardPropagate(eos_widx, eos_fidx);
      for (vector<pair<size_t, vector<size_t>>>::const_iterator it = sentence.begin(); it != sentence.end(); ++it) {
        // train all words even if it is an OOV since <unk> in the vocabulary
        if (!unk_ && it->first == unk_widx) {
          oovcount++;
        } else {
          logp += GetLogProb(it->first, !nce_);
          ivcount++;
        }
        BackPropagate(it->first, it->second);
        if (++bpos == algopts_.batch_size_) {
          FastUpdateWeightsMajor(curr_learning_rate);
          bpos = 0;
        }
        ForwardPropagate(it->first, it->second);
      }
      if (nce_) {
        logp += GetLogProb(eos_widx, false);
      } else {
        logp += GetLogProb(eos_widx, true);
      }
      ivcount++;
      BackPropagate(eos_widx, eos_fidx);

      sents_processed++;
      if ((sents_processed % 500) == 0)
        cout << "." << flush;
    }
    // Do the update for current epoch since the last minibatch.
    FastUpdateWeightsMajor(curr_learning_rate);
    bpos = 0;
    FastUpdateWeightsMinor();

    cout << "\nnum IV words (including </s>) in training: " << ivcount << endl;
    cout << "number of OOV words in training: " << oovcount << endl;
    if (!nce()) {
      cout << "training entropy (base 2): " << -logp / log(2) / ivcount << endl;
      cout << "model perplexity on training: " << exp(-logp / ivcount) << endl;
      cout << "log-likelihood (base e) on training is: " << logp << endl;
    } else {
      cout << "NCE objective value on training is: " << nce_obj_ << endl;
      cout << "un-normalized training entropy (base 2): " << -logp / log(2) / ivcount << endl;
      cout << "unnormalied model perplexity on training: " << exp(-logp / ivcount) << endl;
      cout << "un-normalized log-likelihood (base e) on training is: " << logp << endl;
    }
    cout << "epoch finished" << endl << flush;

    if (!outbase.empty()) {
      if (debug_ > 0) {
        WriteLM(outbase + ".ITER_" + to_string(iteration - 1));
      }
    }

    cout << "----------VALIDATION----------" << endl;
    double curr_logp = EvalLM(validation_data, nce_ppl);
    cout << "log-likelihood (base e) on validation is: " << curr_logp << endl;

    clock_t last_end_time = end_time;
    end_time = clock();
    cout << "time elasped " 
        << static_cast<double>(end_time - last_end_time) / CLOCKS_PER_SEC << " secs for this iteration out of "
        << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " secs in total." << endl;

    if (curr_logp < last_logp) {
      cout << "validation log-likelihood decrease; resetting parameters" << endl;
      RestoreLastParams();
    } else {
      CacheCurrentParams();
    }

    if (curr_logp * algopts_.min_improvement_ <= last_logp) {
      if (!halve_alpha) {
        halve_alpha = true;
      } else {
        if (!outbase.empty()) {
          WriteLM(outbase);
        }
        break;
      }
    }

    if (halve_alpha) {
      curr_learning_rate /= 2;
    }

    last_logp = curr_logp;
  }
}

size_t FNeuralNetLMBase::NCESampleWord(boost::mt19937 &rng_engine) {
  return word_noise_distribution_(rng_engine);
}

size_t FNeuralNetLMBase::NCESampleFactor(boost::mt19937 &rng_engine) {
  assert(weight_factor_output_ > 0);
  return factor_noise_distribution_(rng_engine);
}

void FNeuralNetLMBase::NCECheckSampling() {
  const size_t word_vocab_size = word_vocab_.size();
  size_t draws = 100 * word_vocab_size;
  vector<int> word_samples(word_vocab_size, 0);
  boost::mt19937 rng_engine;
  for (size_t i = 0; i < draws; i++) {
    word_samples[NCESampleWord(rng_engine)]++;
  }

  size_t out = 0;
  for (size_t i = 0; i < word_vocab_size; i++) {
    const double p = word_noise_pdf_[i];
    const double q = 1.0 - p;
    const int expected = int(p * draws);
    const double stddev = sqrt(p * q * draws);
    if (abs(expected - word_samples[i]) > 3*stddev) {
      if (debug_ > 1) {
        cout << word_vocab_.type4idx(i) << " " << "expected: " << expected << "; seen: " << word_samples[i] << endl;
      }
      out++;
    }
  }

  // N.B. It is okay that out > expected_out. Some maths are needed to get an idea about
  // what is needed.
  size_t expected_out = static_cast<size_t>((1.0 - 0.9973) * word_vocab_size);
  cout << "NCE Sampling check: expected " << expected_out 
      << " abnormal counts in word vocabulary of size " << word_vocab_size 
      << "; saw: " << out << endl;

  if (weight_factor_output_ > 0) {
    const size_t factor_vocab_size = factor_vocab_.size();
    draws = 100 * factor_vocab_size;
    vector<int> factor_samples(factor_vocab_size, 0);
    boost::mt19937 rng_engine;
    for (size_t i = 0; i < draws; i++) {
      factor_samples[NCESampleFactor(rng_engine)]++;
    }

    out = 0;
    for (size_t i = 0; i < factor_vocab_size; i++) {
      const double p = factor_noise_pdf_[i];
      const double q = 1.0 - p;
      const int expected = int(p * draws);
      const double stddev = sqrt(p * q * draws);
      if (abs(expected - factor_samples[i]) > 3*stddev) {
        if (debug_ > 1) {
          cout << factor_vocab_.type4idx(i) << " " << "expected: " << expected << "; seen: " << factor_samples[i] << endl;
        }
        out++;
      }
    }

    // N.B. It is okay that out > expected_out. Some maths are needed to get an idea about
    // what is needed.
    expected_out = static_cast<size_t>((1.0 - 0.9973) * factor_vocab_size);
    cout << "NCE Sampling check: expected " << expected_out 
        << " abnormal counts in factor vocabulary of size " << factor_vocab_size 
        << "; saw: " << out << endl;
  }
}

} // namespace fnnlm
