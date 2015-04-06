#ifndef NEURALNET_NEURALNET_LAYER_H_
#define NEURALNET_NEURALNET_LAYER_H_

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include "futil.h"
#include "neuralnet_types.h"
#include "neuralnet_layer_base.h"

namespace neuralnet {

class NeuralNetLayer : public NeuralNetLayerBase {

 public:
  // Constructor.
  NeuralNetLayer() : NeuralNetLayerBase() {}
  // Destructor.
  virtual ~NeuralNetLayer() {}

  // Returns the activation of a neuron.
  virtual ActivationType activations(std::size_t idx) const override {
    assert(ac_state());
    assert(idx < nneurons());
    return activations_[idx];
  }
  // Returns the activations of all neurons.
  const std::vector<ActivationType>& activations() const {
    assert(ac_state());
    return activations_;
  }
  // Sets the activation of a neuron.
  // ac_state_ will be set to true automatically.
  virtual void set_activations(std::size_t idx, ActivationType val) override {
    assert(idx < nneurons());
    activations_[idx] = val;
    set_ac_state(true);
  }

  // Sets the activations of all neurons.
  // ac_state_ will be set to true automatically.
  void set_activations(const std::vector<ActivationType> &ac) {
    assert(ac.size() == nneurons());
    activations_ = ac;
    set_ac_state(true);
  }

  // Sets all activations to value.
  void SetActivationsToValue(ActivationType val) {
    std::fill(activations_.begin(), activations_.end(), val);
    set_ac_state(true);
  }

  // Accumulates the activation by value.
  void AccumulateActivation(std::size_t idx, ActivationType val) {
    assert(ac_state());
    assert(idx < nneurons());
    activations_[idx] += val;
  }

 private:
  // Implementation of AllocateLayer.
  virtual void AllocateLayerImpl() override {
    activations_.resize(nneurons());
    AllocateLayerImplImpl();
  }
  // Implementation of AllocateLayerImpl.
  virtual void AllocateLayerImplImpl() {}
  // Implementation of WriteLayerImpl.
  virtual void WriteLayerImpl(std::ofstream &ofs) override {
    write_1d_vector(ofs, activations_);
    WriteLayerImplImpl(ofs);
  }
  // Implementation of WriteLayerImplImpl.
  virtual void WriteLayerImplImpl(std::ofstream &ofs) {}
  // Implementation of ReadLayerImpl.
  virtual void ReadLayerImpl(std::ifstream &ifs) override {
    read_1d_vector(ifs, activations_);
    ReadLayerImplImpl(ifs);
  }
  // Implementation of ReadLayerImplImpl.
  virtual void ReadLayerImplImpl(std::ifstream &ifs) {}
  // Implentation of ComputeActivation.
  // By default do nothing (for is_input_ = true).
  virtual void ComputeActivationsImpl() override {}
  // Implentation of ComputeError.
  // Notes: save memory allocation time, but violates the encapsulation to some extent.
  // By default do nothing (for is_input_ = true).
  virtual void ComputeErrorsImpl() override {}

  // The activations of the neurons.
  std::vector<ActivationType> activations_;
};

} // namespace neuralnet

#endif
