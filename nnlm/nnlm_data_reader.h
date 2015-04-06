#ifndef NNLM_NNLM_DATA_READER_H_
#define NNLM_NNLM_DATA_READER_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include "nnlm_vocab.h"

namespace nnlm {

// NeurlaNet LM data reader.
class NNLMDataReader {
 public:
  // Constructors.
  NNLMDataReader() :
      shuffle_datafiles_(false), shuffle_sentences_(false) {}
  NNLMDataReader(const std::vector<std::string> &fns, nnlm::NNLMVocab *pv,
                 bool sdata, bool ssent) :
      filenames_(fns), ptr_vocab_(pv),
      shuffle_datafiles_(sdata), shuffle_sentences_(ssent) {}
  // Destructor.
  ~NNLMDataReader() {}

  // Sets the flag shuffle_datafiles_.
  void set_shuffle_datafiles(bool s) { shuffle_datafiles_ = s; }
  // Sets the flag shuffle_sentences_.
  void set_shuffle_sentences(bool s) { shuffle_sentences_ = s; }
  // Sets the filename of the data file.
  void set_filenames(const std::vector<std::string> &fns) { filenames_ = fns; }
  // Sets the pointer to vocabulary.
  void set_ptr_vocab(NNLMVocab *pv) { ptr_vocab_ = pv; }

  // Initializes data reader.
  void StartEpoch();
  // Fills the vector with the indices of the words in the next sentence.
  // Returns false if reaches the end of the file list.
  bool GetSentence(std::vector<std::size_t> &words);

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
  std::vector<std::vector<std::size_t>> sentences_;
  // Iterator pointing to next sentence.
  std::vector<std::vector<std::size_t>>::const_iterator it_next_sentence_;

  // Pointer to the vocabulary.
  NNLMVocab *ptr_vocab_;

  // Random number generator related parameters.
  boost::mt19937 rng_engine_;
  boost::uniform_int<long long> rng_uni_dist_;
};

} // namespace nnlm

#endif
