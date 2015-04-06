#ifndef NEURALNET_NEURALNET_EXP_LAYER_H_
#define NEURALNET_NEURALNET_EXP_LAYER_H_

#include <cstdlib>
#include <iostream>
#include <vector>
#include "neuralnet_layer.h"

namespace neuralnet {

class NeuralNetExpLayer : public NeuralNetLayer {
 public:
  // Constructor.
  NeuralNetExpLayer() {}
  // Destructor.
  virtual ~NeuralNetExpLayer() {}

 private:
  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override {
    std::cerr << "ComputeErrorsImpl of NeuralNetExpLayer not implemented yet!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
};

} // namespace neuralnet


#endif
