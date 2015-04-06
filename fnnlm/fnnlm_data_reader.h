#ifndef FNNLM_FNNLM_DATA_READER_H_
#define FNNLM_FNNLM_DATA_READER_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include "fnnlm_vocab.h"

namespace fnnlm {

// NeurlaNet LM data reader.
class FNNLMDataReader {
 public:
  // Constructors.
  FNNLMDataReader() :
      shuffle_datafiles_(false), shuffle_sentences_(false) {}
  FNNLMDataReader(const std::vector<std::string> &fns, FNNLMVocab *pv_word, FNNLMVocab *pv_factor,
                  bool sdata, bool ssent) :
      filenames_(fns), ptr_word_vocab_(pv_word), ptr_factor_vocab_(pv_factor),
      shuffle_datafiles_(sdata), shuffle_sentences_(ssent) {}
  // Destructor.
  ~FNNLMDataReader() {}

  // Sets the flag shuffle_datafiles_.
  void set_shuffle_datafiles(bool s) { shuffle_datafiles_ = s; }
  // Sets the flag shuffle_sentences_.
  void set_shuffle_sentences(bool s) { shuffle_sentences_ = s; }
  // Sets the filename of the data file.
  void set_filenames(const std::vector<std::string> &fns) { filenames_ = fns; }
  // Sets the pointer to word vocabulary.
  void set_ptr_word_vocab(FNNLMVocab *pv) { ptr_word_vocab_ = pv; }
  // Sets the pointer to factor vocabulary.
  void set_ptr_factor_vocab(FNNLMVocab *pv) { ptr_factor_vocab_ = pv; }

  // Initializes data reader.
  void StartEpoch();
  // Fills the vector with the pairs of (index of the word, indices of the factors) in the next sentence.
  // Returns false if reaches the end of the file list.
  bool GetSentence(std::vector<std::pair<std::size_t, std::vector<std::size_t>>> &sentence);

 private:
  // Load next data file.
  // If reaches end of the file list, return false.
  bool LoadNextDataFile();

  // If true, data files will be shuffled.
  bool shuffle_datafiles_;
  // If true, sentences within the file will be shuffled.
  bool shuffle_sentences_;

  // Filenames of the data files.
  std::vector<std::string> filenames_;
  // Iterator pointing to current data file.
  std::vector<std::string>::const_iterator it_next_filename_;
  // Ifstream for current file.
  std::ifstream ifs_;

  // Sentences.
  std::vector<std::vector<std::pair<std::size_t, std::vector<std::size_t>>>> sentences_;
  // Iterator pointing to next sentence.
  std::vector<std::vector<std::pair<std::size_t, std::vector<std::size_t>>>>::const_iterator it_next_sentence_;

  // Pointer to the word vocabulary.
  FNNLMVocab *ptr_word_vocab_;
  // Pointer to the factor vocabulary.
  FNNLMVocab *ptr_factor_vocab_;

  // Random number generator related parameters.
  boost::mt19937 rng_engine_;
  boost::uniform_int<long long> rng_uni_dist_;
};

} // namespace fnnlm

#endif
