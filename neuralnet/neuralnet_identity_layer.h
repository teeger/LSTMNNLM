#ifndef NEURALNET_NEURALNET_IDENTITY_LAYER_H_
#define NEURALNET_NEURALNET_IDENTITY_LAYER_H_

#include "neuralnet_numeric.h"
#include "neuralnet_layer.h"

namespace neuralnet {

// Neural network identity layer.
class NeuralNetIdentityLayer : public NeuralNetLayer {
 public:
  // Constructor.
  NeuralNetIdentityLayer() {}
  // Destructor.
  ~NeuralNetIdentityLayer() {}

 private:
  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override {
    set_activations(activationinputs());
  }
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override {
    set_errors(errorinputs());
  }

};

} // namespace neuralnet


#endif
