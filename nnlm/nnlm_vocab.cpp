#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ios>
#include <string>
#include <map>
#include "../neuralnet/futil.h"
#include "nnlm_vocab.h"

// <cstdlib>
using std::exit;
// <iostream>
using std::cout;
using std::cerr;
using std::endl;
// <ios>
using std::ios;
// <fstream>
using std::ifstream;
using std::ofstream;
// <string>
using std::string;
using std::getline;
// <map>
using std::map;

namespace nnlm {

void NNLMVocab::ReadVocabFromTxt(const string &vocabtxt) {
  ifstream ifs;
  ifs.open(vocabtxt, ios::in);
  if (ifs.fail()) {
    cerr << "Unable to open " << vocabtxt << endl;
    exit(EXIT_FAILURE);
  }

  string line;
  size_t widx = 0;
  while (getline(ifs, line)) {
    if (idx4word_.find(line) != idx4word_.end()) {
      cerr << "double defined word " << line << " in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
    idx4word_[line] = widx;
    word4idx_.push_back(line);
    widx++;
  }

  if (idx4word_.find("</s>") == idx4word_.end()) {
    cerr << "end-of-sentence </s> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  eos_idx_ = idx4word_["</s>"];
  if (idx4word_.find("<unk>") == idx4word_.end()) {
    cerr << "OOV <unk> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  unk_idx_ = idx4word_["<unk>"];

  size_ = idx4word_.size();
}

void NNLMVocab::ReadVocab(ifstream &ifs) {
  cout << "***reading vocab***" << endl;
  neuralnet::read_single(ifs, size_);
  cout << "size_: " << size_ << endl;
  neuralnet::read_string2T_map(ifs, idx4word_);
  neuralnet::read_1d_string(ifs, word4idx_);

  eos_idx_ = idx4word_["</s>"];
  cout << "eos_idx_: " << eos_idx_ << endl;
  map<string, size_t>::iterator mi = idx4word_.find("<unk>");
  if (mi == idx4word_.end()) {
    cerr << "Internal error: </unk> not in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  } else {
    unk_idx_ = mi->second;
  }
  cout << "unk_idx_: " << unk_idx_ << endl;
}

void NNLMVocab::WriteVocab(ofstream &ofs) {
  neuralnet::write_single(ofs, size_);
  neuralnet::write_string2T_map(ofs, idx4word_);
  neuralnet::write_1d_string(ofs, word4idx_);
}

} // namespace nnlm
