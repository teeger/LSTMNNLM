#ifndef NEURALNET_NEURALNET_CENTERED_SIGMOID_LAYER_H_
#define NEURALNET_NEURALNET_CENTERED_SIGMOID_LAYER_H_

#include "neuralnet_numeric.h"
#include "neuralnet_layer.h"

namespace neuralnet {

class NeuralNetCenteredSigmoidLayer : public NeuralNetLayer {
 public:
  // Constructor.
  // default tanh
  NeuralNetCenteredSigmoidLayer() : scale(2.f) {}
  // Destructor.
  virtual ~NeuralNetCenteredSigmoidLayer() {}

 private:
  // inline compute centered_sigmoid activation function
  ActivationType centered_sigmoid_ac_func(ActivationType val) {
    return (2.0 / (1.0 + stable_fast_exp(-scale * val)) - 1.0);
  }

  // inline compute centered_sigmoid error function
  ErrorType centered_sigmoid_er_func(ErrorType er_val, ActivationType ac_val) {
    return er_val * 2 * scale * ac_val * (1.0 - ac_val);
  }

  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override ;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override;

  // scale
  ActivationType scale;

};

} // namespace neuralnet


#endif
