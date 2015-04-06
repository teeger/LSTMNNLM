#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ios>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <boost/random/variate_generator.hpp>
#include <boost/tokenizer.hpp>
#include "nnlm_data_reader.h"

// <cstdlib>
using std::exit;
// <cstdio>
using std::size_t;
// <iostream>
using std::cerr;
using std::endl;
// <ios>
using std::ios;
// <string>
using std::string;
using std::getline;
// <vector>
using std::vector;
// <map>
using std::map;
// <algorithm>
using std::random_shuffle;
// <boost/random/variate_generator.hpp>
using boost::variate_generator;
// <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;

namespace nnlm {

void NNLMDataReader::StartEpoch() {
  if (filenames_.empty()) {
    cerr << "Filenames_ is empty!" << endl;
    exit(EXIT_FAILURE);
  }
  if (filenames_.size() == 1) {
    // Avoid unnecessary tokenization 
    if (sentences_.empty()) {
      it_next_filename_ = filenames_.begin();
      LoadNextDataFile();
    } else {
      if (shuffle_sentences_) {
        boost::variate_generator<boost::mt19937&, boost::uniform_int<long long>> rng(rng_engine_, rng_uni_dist_);
        random_shuffle(sentences_.begin(), sentences_.end(), rng);
      }
      it_next_sentence_ = sentences_.begin();
    }
  } else {
    if (shuffle_datafiles_) {
      boost::variate_generator<boost::mt19937&, boost::uniform_int<long long>> rng(rng_engine_, rng_uni_dist_);
      random_shuffle(filenames_.begin(), filenames_.end(), rng);
    }
    it_next_filename_ = filenames_.begin();
    LoadNextDataFile();
  }
}

bool NNLMDataReader::GetSentence(vector<size_t> &words) {
  if (it_next_sentence_ == sentences_.end()) {
    if (!LoadNextDataFile()) {
      return false;
    }
  }
  words = *it_next_sentence_;
  ++it_next_sentence_;
  return true;
}

bool NNLMDataReader::LoadNextDataFile() {
  if (it_next_filename_ == filenames_.end()) {
    return false;
  } else {
    // Remember to close the stream.
    ifs_.open(*it_next_filename_, ios::in);
    if (ifs_.fail()) {
      std::cerr << "Unable to open " << *it_next_filename_ << std::endl;
      std::exit(EXIT_FAILURE);
    }

    string line;
    char_separator<char> separator(" ");
    size_t current_idx = 0;
    sentences_.clear();
    while (getline(ifs_, line)) {
      sentences_.push_back(vector<size_t>());
      vector<size_t> &s = sentences_[current_idx++];
      tokenizer<char_separator<char>> tokens(line, separator);
      for (tokenizer<char_separator<char>>::const_iterator it = tokens.begin(); it != tokens.end(); ++it) {
        const size_t w = ptr_vocab_->idx4word(*it);
        if (w == ptr_vocab_->eos_idx()) {
          cerr << "</s> in the middle of the sentence" << endl;
          exit(EXIT_FAILURE);
        }
        s.push_back(w);
      }
      if (s.empty()) {
        cerr << "Empty line in " << *it_next_filename_ << endl;
        exit(EXIT_FAILURE);
      }
    }
    if (shuffle_sentences_) {
      boost::variate_generator<boost::mt19937&, boost::uniform_int<long long>> rng(rng_engine_, rng_uni_dist_);
      random_shuffle(sentences_.begin(), sentences_.end(), rng);
    }
    it_next_sentence_ = sentences_.begin();
    ++it_next_filename_;
    // Remember to close the stream.
    ifs_.close();
    return true;
  }
}


} // namespace nnlm
