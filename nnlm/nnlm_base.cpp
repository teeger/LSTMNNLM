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
#include <limits>
#include "../neuralnet/futil.h"
#include "nnlm_base.h"

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
// <limits>
using std::numeric_limits;

namespace nnlm {

void NeuralNetLMBase::TrainLM(const string &validationfile,
                              const string &outbase,
                              bool nce_ppl) {
  // Prepare for the training
  // Equivalent to ReadLM
  vocab_.ReadVocabFromTxt(vocab_filename_);
  if (vocab_.empty()) {
    cerr << "empty vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  PrintParams();
  CheckParams();
  AllocateModel();
  InitializeNeuralNet();

  // Read the data
  NNLMDataReader train_data(train_filenames_, &vocab_, 
                            shuffle_datafiles_, shuffle_sentences_);
  vector<string> validation_filenames = { validationfile };
  NNLMDataReader validation_data(validation_filenames, &vocab_, false, false);

  // Set NCE sampling.
  if (nce_) {
    // TODO: flatten noise_distribution_?
    vector<int> word_count(vocab_.size(), 0);
    int num_tokens = 0;
     
    const size_t eos_idx = vocab().eos_idx();
    vector<size_t> words;

    train_data.StartEpoch();
    while(train_data.GetSentence(words)) {
      for (size_t p = 0; p < words.size(); p++) {
        word_count[words[p]]++;
        num_tokens++;
      }
      word_count[eos_idx]++;
      num_tokens++;
    }

    noise_distribution_ = Distribution(word_count.begin(), word_count.end());
    noise_pdf_ = noise_distribution_.param().probabilities();
    NCECheckSampling();
    log_num_negative_samples_ = log(num_negative_samples_);
  }

  BatchSGDTrain(train_data, validation_data, outbase, nce_ppl);

  cout << "================================================================================" << endl;
  cout << "Log-likelihood (base e) on validation is: " \
      << EvalLM(validation_data, false) << endl;
}

void NeuralNetLMBase::EvalLM(const string &infile, bool nce_ppl) {
  CheckParams();
  vector<string> filenames = { infile };
  NNLMDataReader data(filenames, &vocab_, false, false);
  cout << "log-likelihood (base e) on " << infile << " is: " << EvalLM(data, nce_ppl) << endl;
}

void NeuralNetLMBase::ReadLM(const string &infile) {
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

  vocab_.ReadVocab(ifs);
  if (vocab_.empty()) {
    cerr << "empty vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }

  ReadLMImpl(ifs);

  // try one more read before check eof
  ifs.get(); 
  if (!ifs.eof()) {
    cout << "Error reading model: should be at the end of the file" << endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
}

void NeuralNetLMBase::WriteLM(const string &outbase) {
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

  vocab_.WriteVocab(ofs);

  WriteLMImpl(ofs);

  ofs.close();
}

void NeuralNetLMBase::ExtractWordInputEmbedding(const string &filename) {
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

void NeuralNetLMBase::ExtractWordOutputEmbedding(const string &filename) {
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

void NeuralNetLMBase::PrintParams() {
  cout << "unk_: " << unk_ << endl;
  cout << "vocab_filename_: " << vocab_filename_ << endl;
  cout << "vocab_.size_: " << vocab_.size() << endl;
  cout << "vocab_.eos_idx_: " << vocab_.eos_idx() << endl;
  cout << "vocab_.unk_idx_: " << vocab_.unk_idx() << endl;
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

  PrintParamsImpl();
}

void NeuralNetLMBase::CheckParams() {
  if (vocab_.empty()) {
    cerr << "vocabulary is empty!" << endl;
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

  CheckParamsImpl();
}

double NeuralNetLMBase::EvalLM(NNLMDataReader &data, bool nce_ppl) {
  CheckParams();

  if (nce_ppl && !nce_) {
    cerr << "nce_ppl == true but nce_ is false!" << endl;
    exit(EXIT_FAILURE);
  }

  const size_t eos_idx = vocab_.eos_idx();
  const size_t unk_idx = vocab_.unk_idx();

  vector<size_t> words;
  double total_logp = 0.0;
  size_t sents_processed = 0;
  size_t ivcount = 0;
  size_t oovcount = 0;

  data.StartEpoch();

  ResetActivations();

  while (data.GetSentence(words)) {
    assert(!words.empty());

    if (independent_) {
      ResetActivations();
    }
    double curr_logp = 0.0;
    ForwardPropagate(eos_idx);
    for (size_t p = 0; p < words.size(); p++) {
      const size_t w = words[p];
      if (!unk_ && w == unk_idx) {
        oovcount++;
      } else {
        curr_logp += GetLogProb(w, !nce_ppl);
        ivcount++;
      }
      ForwardPropagate(w);
    }
    curr_logp += GetLogProb(eos_idx, !nce_ppl);
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

void NeuralNetLMBase::BatchSGDTrain(NNLMDataReader &train_data, NNLMDataReader &validation_data,
                                    const string &outbase, bool nce_ppl) {
  const size_t eos_idx = vocab_.eos_idx();
  const size_t unk_idx = vocab_.unk_idx();

  vector<size_t> words;

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
    // NOTE: the vector "words" does not include </s> at the end!
    while (train_data.GetSentence(words)) {
      assert(!words.empty());
      
      if (independent_) {
        ResetActivations();
      }
      ForwardPropagate(eos_idx);
      for (size_t p = 0; p < words.size(); p++) {
        // train all words even if it is an OOV since <unk> in the vocabulary
        const size_t w = words[p];
        if (!unk_ && w == unk_idx) {
          oovcount++;
        } else {
          logp += GetLogProb(w, !nce_);
          ivcount++;
        }
        BackPropagate(w);
        if (++bpos == algopts_.batch_size_) {
          FastUpdateWeightsMajor(curr_learning_rate);
          bpos = 0;
        }
        ForwardPropagate(w);
      }
      if (nce_) {
        logp += GetLogProb(eos_idx, false);
      } else {
        logp += GetLogProb(eos_idx, true);
      }
      ivcount++;
      BackPropagate(eos_idx);

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
      cout << "NCE objective value on training: " << nce_obj_ << endl;
      cout << "un-normalized training entropy (base 2): " << -logp / log(2) / ivcount << endl;
      cout << "unnormalied model perplexity on training: " << exp(-logp / ivcount) << endl;
      cout << "un-normalized log-likelihood (base e) on training: " << logp << endl;
    }
    cout << "num train IV words (including </s>): " << ivcount << endl;
    cout << "epoch finished" << endl << flush;

    if (!outbase.empty()) {
      if (debug_ > 0) {
        WriteLM(outbase + ".ITER_" + to_string(iteration - 1));
      }
    }

    cout << "----------VALIDATION----------" << endl;
    double curr_logp = EvalLM(validation_data, nce_ppl);
    cout << "log-likelihood (base e) on validation: " << curr_logp << endl;

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

size_t NeuralNetLMBase::NCESampleWord(boost::mt19937 &rng_engine) {
  return noise_distribution_(rng_engine);
}

void NeuralNetLMBase::NCECheckSampling() {
  const size_t vocab_size = vocab_.size();
  const size_t draws = 100 * vocab_size;
	vector<int> samples(vocab_size, 0);
  boost::mt19937 rng_engine;
	for (size_t i = 0; i < draws; i++) {
		samples[NCESampleWord(rng_engine)]++;
  }

	size_t out = 0;
	for (size_t i = 0; i < vocab_size; i++) {
		const double p = noise_pdf_[i];
		const double q = 1.0 - p;
		const int expected = int(p * draws);
		const double stddev = sqrt(p * q * draws);
		if (abs(expected - samples[i]) > 3*stddev) {
			if (debug_ > 1) {
        cout << vocab_.word4idx(i) << " " << "expected: " << expected << "; seen: " << samples[i] << endl;
      }
			out++;
		}
	}

  // N.B. It is okay that out > expected_out. Some maths are needed to get an idea about
  // what is needed.
	size_t expected_out = static_cast<size_t>((1.0 - 0.9973) * vocab_size);
  cout << "NCE Sampling check: expected " << expected_out 
      << " abnormal counts in vocabulary of size " << vocab_size 
      << "; saw: " << out << endl;
}

} // namespace nnlm
