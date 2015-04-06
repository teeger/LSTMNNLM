#ifndef NEURALNET_NEURALNET_SIGMOID_LAYER_H_
#define NEURALNET_NEURALNET_SIGMOID_LAYER_H_

#include "neuralnet_numeric.h"
#include "neuralnet_layer.h"

namespace neuralnet {

class NeuralNetSigmoidLayer : public NeuralNetLayer {
 public:
  // Constructor.
  NeuralNetSigmoidLayer() {}
  // Destructor.
  virtual ~NeuralNetSigmoidLayer() {}

 private:
  // inline compute sigmoid activation function
  ActivationType sigmoid_ac_func(ActivationType val) {
    return (1.0 / (1.0 + stable_fast_exp(-val)));
  }

  // inline compute sigmoid error function
  ErrorType sigmoid_er_func(ErrorType er_val, ActivationType ac_val) {
    return er_val * ac_val * (1.0 - ac_val);
  }

  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() override ;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() override;
};

} // namespace neuralnet


#endif
