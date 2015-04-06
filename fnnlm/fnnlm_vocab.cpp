#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ios>
#include <string>
#include <map>
#include "../neuralnet/futil.h"
#include "fnnlm_vocab.h"

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

namespace fnnlm {

void FNNLMVocab::ReadVocabFromTxt(const string &vocabtxt) {
  ifstream ifs;
  ifs.open(vocabtxt, ios::in);
  if (ifs.fail()) {
    cerr << "Unable to open " << vocabtxt << endl;
    exit(EXIT_FAILURE);
  }

  string line;
  size_t widx = 0;
  while (getline(ifs, line)) {
    if (idx4type_.find(line) != idx4type_.end()) {
      cerr << "double defined type " << line << " in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
    idx4type_[line] = widx;
    type4idx_.push_back(line);
    widx++;
  }

  if (idx4type_.find("</s>") == idx4type_.end()) {
    cerr << "end-of-sentence </s> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  eos_idx_ = idx4type_["</s>"];
  if (idx4type_.find("<unk>") == idx4type_.end()) {
    cerr << "OOV <unk> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  unk_idx_ = idx4type_["<unk>"];

  size_ = idx4type_.size();
}

void FNNLMVocab::ReadVocab(ifstream &ifs) {
  cout << "***reading vocab***" << endl;
  neuralnet::read_single(ifs, size_);
  cout << "size_: " << size_ << endl;
  neuralnet::read_string2T_map(ifs, idx4type_);
  neuralnet::read_1d_string(ifs, type4idx_);

  eos_idx_ = idx4type_["</s>"];
  cout << "eos_idx_: " << eos_idx_ << endl;
  map<string, size_t>::iterator mi = idx4type_.find("<unk>");
  if (mi == idx4type_.end()) {
    cerr << "Internal error: </unk> not in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  } else {
    unk_idx_ = mi->second;
  }
  cout << "unk_idx_: " << unk_idx_ << endl;
}

void FNNLMVocab::WriteVocab(ofstream &ofs) {
  neuralnet::write_single(ofs, size_);
  neuralnet::write_string2T_map(ofs, idx4type_);
  neuralnet::write_1d_string(ofs, type4idx_);
}

} // namespace fnnlm
