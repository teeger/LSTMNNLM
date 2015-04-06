#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "futil.h"

// <cstdio>
using std::size_t;
// <cstdlib>
using std::exit;
// <iostream>
using std::cerr;
using std::endl;
// <fstream>
using std::ifstream;
using std::ofstream;
// <string>
using std::string;
// <vector>
using std::vector;

namespace neuralnet {

void write_string(ofstream &ofs, string &s) {
  size_t sz = s.length() + 1;  // include terminal \0
  ofs.write(reinterpret_cast<char*>(&sz), sizeof(size_t));
  if (ofs.fail()) {
    cerr << "Error writing string" << endl;
    exit(EXIT_FAILURE);
  }
  ofs.write(s.c_str(), sizeof(char) * sz);
  if (ofs.fail()) {
    cerr << "Error writing string" << endl;
    exit(EXIT_FAILURE);
  }
}

void read_string(ifstream &ifs, string &s) {
  size_t sz;
  ifs.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
  if (ifs.fail()) {
    cerr << "Error reading string" << endl;
    exit(EXIT_FAILURE);
  }
  char *buff = new char[sz+2];
  ifs.read(reinterpret_cast<char*>(buff), sizeof(char) * sz);
  if (ifs.fail()) {
    cerr << "Error reading string" << endl;
    exit(EXIT_FAILURE);
  }
  s = string(buff);
  delete[] buff;
}

void write_1d_string(ofstream &ofs, vector<string> &V) {
  size_t s = V.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing 1d string" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (vector<string>::iterator it = V.begin(); it != V.end(); ++it) {
    write_string(ofs, *it);
  }
}

void read_1d_string(ifstream &ifs, vector<string> &V) {
  size_t s = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading 1d string" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  V.resize(s);
  for (vector<string>::iterator it = V.begin(); it != V.end(); ++it) {
    read_string(ifs, *it);
  }
}

} // namespace neuralnet
