#ifndef FNNLM_FNNLM_VOCAB_H_
#define FNNLM_FNNLM_VOCAB_H_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "../neuralnet/futil.h"

namespace fnnlm {

// F-NNLM vocabulary.
class FNNLMVocab {
 public:
  // Constructors.
  FNNLMVocab() : size_(0), unk_idx_(0), eos_idx_(0) {}
  // Destructor.
  ~FNNLMVocab() {}

  // Returns whether vocabulary is empty.
  bool empty() const { return size_ == 0; }
  // Returns number of types.
  std::size_t size() const { return size_; }
  // Returns the index for type s.
  // If it is an OOV, returns unk_idx_.
  std::size_t idx4type(const std::string &s) const {
    assert(size_ > 0);
    std::map<std::string, std::size_t>::const_iterator mi = idx4type_.find(s);
    if (mi == idx4type_.end()) {
      return unk_idx_;
    } else {
      return mi->second;
    }
  }
  // Returns the type for index i.
  const std::string& type4idx(const std::size_t i) const {
    assert(size_ > 0);
    return type4idx_.at(i);
  }
  // Returns the eos_idx_.
  std::size_t eos_idx() const { return eos_idx_; }
  // Returns the unk_idx_.
  std::size_t unk_idx() const { return unk_idx_; }

  // Reads the vocabulary from txt file.
  // Each line is a type.
  void ReadVocabFromTxt(const std::string &vocabtxt);

  // Reads the vocabulary from the binary ifstream.
  void ReadVocab(std::ifstream &ifs);
  // Writes the vocabulary into the binary ofstream.
  void WriteVocab(std::ofstream &ofs);

 private:
  // Number of types in the vocabulary.
  // Including </s> and <unk>.
  std::size_t size_;
  
  // Index for type.
  std::map<std::string, std::size_t> idx4type_;
  // Word for index.
  std::vector<std::string> type4idx_;
  // Indices of </s> and <unk>.
  std::size_t eos_idx_, unk_idx_;
};

} // namespace fnnlm

#endif
