#ifndef NEURALNET_FUTIL_H_
#define NEURALNET_FUTIL_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace neuralnet {

template <class T>
void write_single(std::ofstream &ofs, T v) {
  ofs.write(reinterpret_cast<char*>(&v), sizeof(T));
  if (ofs.fail()) {
    std::cerr << "Error writing single element" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
void read_single(std::ifstream &ifs, T &v) {
  ifs.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (ifs.fail()) {
    std::cerr << "Error reading single element" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Note: cannot be used for writing 1d vector of string!
template <class T>
void write_1d_vector(std::ofstream &ofs, std::vector<T> &V) {
  std::size_t s = V.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing 1d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (s) {
    ofs.write(reinterpret_cast<char*>(&V[0]), sizeof(T) * s);
    if (ofs.fail()) {
      std::cerr << "Error writing 1d vector" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

// Note: cannot be used for reading 1d vector of string!
template <class T>
void read_1d_vector(std::ifstream &ifs, std::vector<T> &V) {
  std::size_t s = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading 1d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  V.resize(s);
  if (s) {
    ifs.read(reinterpret_cast<char*>(&V[0]), sizeof(T) * s);
    if (ifs.fail()) {
      std::cerr << "Error reading 1d vector" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

template <class T>
void write_2d_vector(std::ofstream &ofs, std::vector<std::vector<T> > &V2) {
  std::size_t s = V2.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing 2d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < s; i++) {
    write_1d_vector(ofs, V2[i]);
  }
}

template <class T>
void read_2d_vector(std::ifstream &ifs, std::vector<std::vector<T> > &V2) {
  std::size_t s = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading 2d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  V2.resize(s);
  for (std::size_t i = 0; i < s; i++) {
    read_1d_vector(ifs, V2[i]);
  }
}

template <class T>
void write_3d_vector(std::ofstream &ofs, std::vector<std::vector<std::vector<T>>> &V3) {
  std::size_t s = V3.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing 3d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < s; i++) {
    write_2d_vector(ofs, V3[i]);
  }
}

template <class T>
void read_3d_vector(std::ifstream &ifs, std::vector<std::vector<std::vector<T>>> &V3) {
  std::size_t s = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading 3d vector" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  V3.resize(s);
  for (std::size_t i = 0; i < s; i++) {
    read_2d_vector(ifs, V3[i]);
  }
}

template <class T>
void write_unorderedset(std::ofstream &ofs, std::unordered_set<T> &US) {
  std::size_t s = US.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing unordered_set" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (typename std::unordered_set<T>::iterator si = US.begin(); si != US.end(); ++si) {
    T entry = *si;
    ofs.write(reinterpret_cast<char*>(&entry), sizeof(T));
    if (ofs.fail()) {
      std::cerr << "Error writing unordered_set" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

template <class T>
void read_unorderedset(std::ifstream &ifs, std::unordered_set<T> &US) {
  std::size_t s = 0;
  US.clear();
  ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading unordered_set" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (std::size_t i = 0; i < s; i++) {
    T entry;
    ifs.read(reinterpret_cast<char*>(&entry), sizeof(T));
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_set" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    US.insert(entry);
  }
}

template <class T1, class T2>
void write_unorderedmap(std::ofstream &ofs, std::unordered_map<T1, T2> &UM) {
  std::size_t s = UM.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing unordered_map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (typename std::unordered_map<T1, T2>::iterator mi = UM.begin(); mi != UM.end(); ++mi) {
    T1 s1 = mi->first;
    ofs.write(reinterpret_cast<char*>(&s1), sizeof(T1));
    if (ofs.fail()) {
      std::cerr << "Error writing unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    T2 s2 = mi->second;
    ofs.write(reinterpret_cast<char*>(&s2), sizeof(T2));
    if (ofs.fail()) {
      std::cerr << "Error writing unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

template <class T1, class T2>
void read_unorderedmap(std::ifstream &ifs, std::unordered_map<T1, T2> &UM) {
  UM.clear();
  std::size_t sz;
  ifs.read(reinterpret_cast<char*>(&sz), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading unordered_map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < sz; i++) {
    T1 s1;
    ifs.read(reinterpret_cast<char*>(&s1), sizeof(T1));
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    T2 s2;
    ifs.read(reinterpret_cast<char*>(&s2), sizeof(T2));
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    UM[s1] = s2;
  }
}


template <class T>
void write_string2T_map(std::ofstream &ofs, std::map<std::string, T> &M) {
  std::size_t s = M.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing string2T map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (typename std::map<std::string, T>::iterator mi = M.begin(); mi != M.end(); ++mi) {
    s = std::strlen(mi->first.c_str()) + 1;  // include terminal \0
    ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
    if (ofs.fail()) {
      std::cerr << "Error writing string2T map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ofs.write(mi->first.c_str(), sizeof(char) * s);
    if (ofs.fail()) {
      std::cerr << "Error writing string2T map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    T st = mi->second;
    ofs.write(reinterpret_cast<char*>(&st), sizeof(T));
    if (ofs.fail()) {
      std::cerr << "Error writing string2T map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

template <class T>
void read_string2T_map(std::ifstream &ifs, std::map<std::string, T> &M) {
  M.clear();
  std::size_t sz;
  ifs.read(reinterpret_cast<char*>(&sz), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading unordered_map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int slen = 1000;
  char buff[slen];
  for (std::size_t i = 0; i < sz; i++) {
    std::size_t s;
    ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ifs.read(buff, sizeof(char) * s);
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    T st;
    ifs.read(reinterpret_cast<char*>(&st), sizeof(T));
    if (ifs.fail()) {
      std::cerr << "Error reading unordered_map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    M[std::string(buff)] = st;
  }
}

template <class T>
void write_T2string_map(std::ofstream &ofs, std::map<T, std::string> &M) {
  std::size_t s = M.size();
  ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
  if (ofs.fail()) {
    std::cerr << "Error writing T2string map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (typename std::map<T, std::string>::iterator mi = M.begin(); mi != M.end(); mi++) {
    T st = mi->first;
    ofs.write(reinterpret_cast<char*>(&st), sizeof(T));
    if (ofs.fail()) {
      std::cerr << "Error writing T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    s = std::strlen(mi->second.c_str()) + 1;  // include terminal \0
    ofs.write(reinterpret_cast<char*>(&s), sizeof(std::size_t));
    if (ofs.fail()) {
      std::cerr << "Error writing T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ofs.write(mi->second.c_str(), sizeof(char) * s);
    if (ofs.fail()) {
      std::cerr << "Error writing T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

template <class T>
void read_T2string_map(std::ifstream &ifs, std::map<T, std::string> &M) {
  M.clear();
  std::size_t sz;
  ifs.read(reinterpret_cast<char*>(&sz), sizeof(std::size_t));
  if (ifs.fail()) {
    std::cerr << "Error reading T2string map" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int slen = 1000;
  char buff[slen];
  for (std::size_t i = 0; i < sz; i++) {
    T st;
    std::size_t s;
    ifs.read(reinterpret_cast<char*>(&st), sizeof(T));
    if (ifs.fail()) {
      std::cerr << "Error reading T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ifs.read(reinterpret_cast<char*>(&s), sizeof(std::size_t));
    if (ifs.fail()) {
      std::cerr << "Error reading T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ifs.read(buff, sizeof(char) * s);
    if (ifs.fail()) {
      std::cerr << "Error reading T2string map" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    M[st] = std::string(buff);
  }
}


void write_string(std::ofstream &ofs, std::string &s);
void read_string(std::ifstream &ifs, std::string &s);
void write_1d_string(std::ofstream &ofs, std::vector<std::string> &V);
void read_1d_string(std::ifstream &ifs, std::vector<std::string> &V);

} // namespace neuralnet

#endif
