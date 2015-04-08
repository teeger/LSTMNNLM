#ifndef NEURALNET_NEURALNET_TANH_LAYER_H_
#define NEURALNET_NEURALNET_TANH_LAYER_H_

#include "neuralnet_numeric.h"
#include "neuralnet_layer.h"

namespace neuralnet {

class NeuralNetTanhLayer : public NeuralNetLayer {
 public:
  // Constructor.
  // default tanh
  NeuralNetTanhLayer() {}
  // Destructor.
  virtual ~NeuralNetTanhLayer() {}

 private:
  // inline compute tanh activation function
  ActivationType tanh_ac_func(ActivationType val) {
    return (2.f / (1.f + stable_fast_exp(-2.f * val)) - 1.f);
  }

  // inline compute tanh error function
  ErrorType tanh_er_func(ErrorType er_val, ActivationType ac_val) {
    return er_val * (1.f + ac_val) * (1.f - ac_val);
  }

  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override ;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override;

};

} // namespace neuralnet


#endif
