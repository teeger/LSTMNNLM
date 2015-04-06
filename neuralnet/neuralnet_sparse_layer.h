#ifndef NEURALNET_NEURALNET_SPARSE_LAYER_H_
#define NEURALNET_NEURALNET_SPARSE_LAYER_H_

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "futil.h"
#include "neuralnet_types.h"
#include "neuralnet_layer_base.h"

namespace neuralnet {

class NeuralNetSparseLayer : public NeuralNetLayerBase {

 public:
  // Constructor.
  NeuralNetSparseLayer() {}
  // Destructor.
  virtual ~NeuralNetSparseLayer() {}

  // Returns the activation of a neuron.
  virtual ActivationType activations(std::size_t idx) const override {
    assert(ac_state());
    assert(idx < nneurons());
    std::unordered_map<std::size_t, ActivationType>::const_iterator it = activations_.find(idx);
    if (it != activations_.end()) {
      return it->second;
    } else {
      return 0;
    }
  }
  // Returns the activations of all neurons.
  const std::unordered_map<std::size_t, ActivationType>& activations() const {
    assert(ac_state());
    return activations_;
  }

  // Sets the activation of a neuron.
  // ac_state_ will be set to true automatically.
  virtual void set_activations(std::size_t idx, ActivationType val) override {
    assert(idx < nneurons());
    if (val == 0) {
      std::unordered_map<std::size_t, ActivationType>::const_iterator it = activations_.find(idx);
      if (it != activations_.end()) {
        activations_.erase(it);
      }
    } else {
      activations_[idx] = val;
    }
    set_ac_state(true);
  }
  // Sets the activations of all neurons.
  // ac_state_ will be set to true automatically.
  void set_activations(const std::unordered_map<std::size_t, ActivationType> &ac) {
    activations_ = ac;
    set_ac_state(true);
  }

  // Sets all activations to value.
  void SetActivationsToValue(ActivationType val) {
    if (val == 0) {
      activations_.clear();
    } else {
      for (std::size_t i = 0; i < nneurons(); i++) {
        activations_[i] = val;
      }
    }
    set_ac_state(true);
  }

  // Accumulates the activation by value.
  void AccumulateActivation(std::size_t idx, ActivationType val) {
    assert(ac_state());
    assert(idx < nneurons());
    if (val != 0) {
      std::unordered_map<std::size_t, ActivationType>::iterator it = activations_.find(idx);
      if (it != activations_.end()) {
        if (it->second == -val) {
          activations_.erase(it);
        } else {
          it->second += val;
        }
      } else {
        activations_[idx] += val;
      }
    }
  }

 private:
  // Implementation of WriteLayerImpl.
  virtual void WriteLayerImpl(std::ofstream &ofs) override {
    write_unorderedmap(ofs, activations_);
    WriteLayerImplImpl(ofs);
  }
  // Implementation of WriteLayerImplImpl.
  virtual void WriteLayerImplImpl(std::ofstream &ofs) {}
  // Implementation of ReadLayerImpl.
  void ReadLayerImpl(std::ifstream &ifs) override {
    read_unorderedmap(ifs, activations_);
    ReadLayerImplImpl(ifs);
  }
  // Implementation of ReadLayerImplImpl.
  virtual void ReadLayerImplImpl(std::ifstream &ifs) {}
  // Implentation of ComputeActivation.
  // By default do nothing (for is_input_ = true).
  virtual void ComputeActivationsImpl() override {}
  // Implentation of ComputeError.
  // By default do nothing (for is_input_ = true).
  virtual void ComputeErrorsImpl() override {}

  // The activations of the neurons.
  std::unordered_map<std::size_t, ActivationType> activations_;
};

} // namespace neuralnet

#endif
