#ifndef NEURALNET_NEURALNET_MAP_CONNECTION_H_
#define NEURALNET_NEURALNET_MAP_CONNECTION_H_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "futil.h"
#include "neuralnet_layer_base.h"
#include "neuralnet_layer.h"
#include "neuralnet_sparse_layer.h"

namespace neuralnet {

// [Normal Usage]
// map connection no weight, just copy
// the input activations directly to the output
class NeuralNetMapConnection {

 public:
  // Constructor.
  NeuralNetMapConnection() : nneurons_(0) {};
  // Destructor.
  virtual ~NeuralNetMapConnection() {};


  void set_nneurons(std::size_t nn) { 
    nneurons_ = nn;
    //AllocateConnection();
  }
    
  
  void ForwardPropagate(const NeuralNetLayer &input, NeuralNetLayerBase &output);

  void BackPropagate(const NeuralNetLayerBase &output, NeuralNetLayerBase &input);

  // Writes the connection to stream.
  void WriteConnection(std::ofstream &ofs) {
    write_single(ofs, nneurons_);
  }
  // Reads the connection from stream.
  void ReadConnection(std::ifstream &ifs) {
    std::cout << "***reading connection***" << std::endl;
    read_single(ifs, nneurons_);
    std::cout << "nneurons_: " << nneurons_ << std::endl;
  }

 private:
  // The number of inputs and outputs of the connection.
  std::size_t nneurons_;

};

} // namespace neuralnet

#endif
