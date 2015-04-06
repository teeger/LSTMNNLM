#ifndef NEURALNET_NEURALNET_SOFTMAX_LAYER_H_
#define NEURALNET_NEURALNET_SOFTMAX_LAYER_H_

#include <cstdlib>
#include <iostream>
#include <vector>
#include "neuralnet_numeric.h"
#include "neuralnet_layer.h"

namespace neuralnet {

class NeuralNetSoftmaxLayer : public NeuralNetLayer {
 public:
  // Constructor.
  NeuralNetSoftmaxLayer() {}
  // Destructor.
  virtual ~NeuralNetSoftmaxLayer() {}

 private:
  // Implementation of AllocateLayerImpl.
  virtual void AllocateLayerImplImpl() override {
    aux_activations_.resize(nneurons());
  }

  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override {
    std::cerr << "ComputeErrorsImpl of NeuralNetSoftmaxLayer not implemented yet!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Auxilary vector for computing activations.
  // Avoid dynamically allocating memory.
  std::vector<ActivationType> aux_activations_;

};

} // namespace neuralnet


#endif
