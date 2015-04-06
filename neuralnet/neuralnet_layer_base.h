#ifndef NEURALNET_NEURALNET_LAYER_BASE_H_
#define NEURALNET_NEURALNET_LAYER_BASE_H_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <vector>
#include "futil.h"
#include "neuralnet_types.h"

namespace neuralnet {

// Base class of neural network layers.
// 1. If is_input_ == true, then no activationinputs_, errors_, errorinputs_.
// 2. If is_output_ == true, then no errorinputs_.
// Note: We assume the developer carefully set the flags is_input_ and
// is_output_, while not calling the related functions. We only do the sanity
// check under debug mode (#define NDEBUG), not under release mode.
class NeuralNetLayerBase {
 public:
  // Constructor.
  NeuralNetLayerBase() : is_input_(false), is_output_(false),
    nneurons_(0), errorinput_cutoff_(0),
    ac_state_(false), er_state_(false) {}
  // Destructor.
  virtual ~NeuralNetLayerBase() {}

  // Returns the flag is_input_.
  bool is_input() const { return is_input_; }
  // Returns the flag is_output_.
  bool is_output() const { return is_output_; }
  // Returns the number of neurons.
  std::size_t nneurons() const { return nneurons_; }
  // Returns the cutoff of the input of neuron errors. 
  ErrorType errorinput_cutoff() const { return errorinput_cutoff_; }
  // Returns the state of neuron activations.
  bool ac_state() const { return ac_state_; }
  // Returns the input of a neuron activation.
  ActivationInputType activationinputs(std::size_t idx) const {
    assert(!is_input_);
    assert(!ac_state_);
    assert(idx < nneurons_);
    return activationinputs_[idx];
  }
  // Returns the inputs of all neuron activations.
  const std::vector<ActivationInputType>& activationinputs() const {
    assert(!is_input_);
    assert(!ac_state_);
    return activationinputs_;
  }
  // Returns the state of neuron errors.
  bool er_state() const { return er_state_; }
  // Returns the input of a neuron error. 
  ErrorInputType errorinputs(std::size_t idx) const {
    assert(!is_input_);
    assert(!is_output_);
    assert(!er_state_);
    assert(idx < nneurons_);
    return errorinputs_[idx];
  }
  // Returns the inputs of all neuron errors. 
  const std::vector<ErrorInputType>& errorinputs() const {
    assert(!is_input_);
    assert(!is_output_);
    assert(!er_state_);
    return errorinputs_;
  }
  // Returns the activation of a neuron.
  virtual ActivationType activations(std::size_t idx) const = 0;
  // Returns the error of a neuron.
  ErrorType errors(std::size_t idx) const {
    assert(!is_input_);
    assert(er_state_);
    assert(idx < nneurons_);
    return errors_[idx];
  }
  // Returns the errors of all neurons.
  const std::vector<ErrorType>& errors() const {
    assert(!is_input_);
    assert(er_state_);
    return errors_;
  }

  // Sets the number of neurons, flags is_input_ and is_output_.
  // Automatically calls AllocateLayer().
  void set_nneurons(std::size_t n, bool input, bool output) { 
    nneurons_ = n; 
    is_input_ = input;
    is_output_ = output;
    if (is_input_ && is_output_) {
      std::cerr << "is_input_ and is_output_ cannot both be true" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    AllocateLayer();
  }
  // Sets the cutoff of the input of neuron errors.
  void set_errorinput_cutoff(ErrorInputType c) { 
    assert(!is_input_);
    assert(!is_output_);
    errorinput_cutoff_ = c; 
  }
  // Sets the state of neuron activations.
  void set_ac_state(bool state) { ac_state_ = state; }
  // Sets the input of a neuron activation.
  // ac_state_ will be set to false automatically.
  void set_activationinputs(std::size_t idx, ActivationInputType val) {
    assert(!is_input_);
    assert(idx < nneurons_);
    activationinputs_[idx] = val;
    ac_state_ = false;
  }
  // Sets the inputs of all neuron activations.
  // ac_state_ will be set to false automatically.
  void set_activationinputs(const std::vector<ActivationInputType> &acin) {
    assert(!is_input_);
    activationinputs_ = acin;
    ac_state_ = false;
  }
  // Sets the state of neuron errors.
  void set_er_state(bool state) { er_state_ = state; }
  // Sets the input of a neuron error.
  // er_state_ will be set to false automatically.
  void set_errorinputs(std::size_t idx, ErrorInputType val) {
    assert(!is_input_);
    assert(!is_output_);
    assert(idx < nneurons_);
    errorinputs_[idx] = val;
    er_state_ = false;
  }
  // Sets the inputs of all neuron inputs.
  // er_state_ will be set to false automatically.
  void set_errorinputs(const std::vector<ErrorInputType> &erin) {
    assert(!is_input_);
    assert(!is_output_);
    errorinputs_ = erin;
    er_state_ = false;
  }
  // Sets the activation of a neuron.
  virtual void set_activations(std::size_t idx, ActivationType val) = 0;
  // Sets the error of a neuron.
  void set_errors(std::size_t idx, ErrorType val) {
    assert(!is_input_);
    assert(idx < nneurons_);
    errors_[idx] = val;
    er_state_ = true;
  }
  // Sets the errors of all neurons.
  void set_errors(const std::vector<ErrorType> &er) {
    assert(!is_input_);
    assert(er.size() == nneurons_);
    errors_ = er;
    er_state_ = true;
  }
  
  // Resets the input of all neuron activations to 0, and ac_state_ to false.
  void ResetInputForActivations() {
    assert(!is_input_);
    std::fill(activationinputs_.begin(), activationinputs_.end(), 0.0f);
    ac_state_ = false;
  }
  // Resets the input of the specified neuron activation to 0, and ac_state_ to false.
  void ResetInputForActivation(std::size_t idx) {
    assert(!is_input_);
    assert(idx < nneurons_);
    activationinputs_[idx] = 0;
    ac_state_ = false;
  }
  // Resets the input of all neuron errors to 0, and er_state_ to false.
  void ResetInputForErrors() {
    assert(!is_input_);
    assert(!is_output_);
    std::fill(errorinputs_.begin(), errorinputs_.end(), 0.0f);
    er_state_ = false;
  }
  // Resets the input of the specified neuron error to 0, and er_state_ to false.
  void ResetInputForError(std::size_t idx) {
    assert(!is_input_);
    assert(!is_output_);
    assert(idx < nneurons_);
    errorinputs_[idx] = 0;
    er_state_ = false;
  }

  // Accumulates the weighted sum input for activation of the specified neuron.
  void AccumulateInputForActivation(std::size_t idx, ActivationInputType val) {
    assert(!is_input_);
    assert(!ac_state_);
    assert(idx < nneurons_);
    activationinputs_[idx] += val;
  }
  // Accumulates the weighted sum input for error of the specified neuron.
  void AccumulateInputForError(std::size_t idx, ErrorInputType val) {
    assert(!is_input_);
    assert(!is_output_);
    assert(!er_state_);
    assert(idx < nneurons_);
    ErrorType &erin = errorinputs_[idx];
    erin += val;
    if (errorinput_cutoff_ > 0) {
      if (erin > errorinput_cutoff_) {
        erin = errorinput_cutoff_;
      }
      if (erin < -errorinput_cutoff_) {
        erin = -errorinput_cutoff_;
      }
    }
  }

  // Sets all errors to value.
  void SetErrorsToValue(ErrorType val) {
    assert(!is_input_);
    std::fill(errors_.begin(), errors_.end(), val);
    er_state_ = true;
  }

  // Accumulates the error by value.
  void AccumulateError(std::size_t idx, ErrorType val) {
    assert(!is_input_);
    assert(er_state());
    assert(idx < nneurons());
    errors_[idx] += val;
  }

  // Computes all neuron activations according to their inputs.
  void ComputeActivations() {
    assert(!is_input_);
    assert(!ac_state_);
    ComputeActivationsImpl();
    ac_state_ = true;
  }
  // Computes all neuron errors according to their inputs.
  void ComputeErrors() {
    assert(!is_input_);
    assert(!is_output_);
    assert(!er_state_);
    assert(ac_state_);
    ComputeErrorsImpl();
    er_state_ = true;
  }

  // Writes the layer to stream.
  void WriteLayer(std::ofstream &ofs) {
    write_single(ofs, is_input_);
    write_single(ofs, is_output_);
    write_single(ofs, nneurons_);
    write_single(ofs, ac_state_);
    write_single(ofs, er_state_);
    write_single(ofs, errorinput_cutoff_);

    if (!is_input_) {
      write_1d_vector(ofs, activationinputs_);

      if (!is_output_) {
        write_1d_vector(ofs, errorinputs_);
      }
      write_1d_vector(ofs, errors_);
    }

    WriteLayerImpl(ofs);
  }
  // Reads the layer from stream.
  void ReadLayer(std::ifstream &ifs) {
    std::cout << "***reading layer***" << std::endl;
    read_single(ifs, is_input_);
    std::cout << "is_inputs_: " << is_input_ << std::endl;
    read_single(ifs, is_output_);
    std::cout << "is_outputs_: " << is_output_ << std::endl;
    read_single(ifs, nneurons_);
    std::cout << "nneurons_: " << nneurons_ << std::endl;
    read_single(ifs, ac_state_);
    std::cout << "ac_state_: " << ac_state_ << std::endl;
    read_single(ifs, er_state_);
    std::cout << "er_state_: " << er_state_ << std::endl;
    read_single(ifs, errorinput_cutoff_);
    std::cout << "errorinput_cutoff_: " << errorinput_cutoff_ << std::endl;

    if (!is_input_) {
      read_1d_vector(ifs, activationinputs_);

      if (!is_output_) {
        read_1d_vector(ifs, errorinputs_);
      }
      read_1d_vector(ifs, errors_);
    }

    ReadLayerImpl(ifs);
  }

 private:
  // Allocates the layer.
  void AllocateLayer() {
    if (nneurons_ == 0) {
      std::cerr << "nneurons_ should be greater than 0!" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if (!is_input_) {
      activationinputs_.resize(nneurons_);

      if (!is_output_) {
        errorinputs_.resize(nneurons_);
      }
      errors_.resize(nneurons_);
    }

    AllocateLayerImpl();
  };
  // Implementation of AllocateLayer.
  virtual void AllocateLayerImpl() {}
  // Implementation of WriteLayerImpl.
  virtual void WriteLayerImpl(std::ofstream &ofs) {}
  // Implementation of ReadLayerImpl.
  virtual void ReadLayerImpl(std::ifstream &ifs) {}
  // Implentation of ComputeActivation.
  virtual void ComputeActivationsImpl() = 0;
  // Implentation of ComputeError.
  virtual void ComputeErrorsImpl() = 0;

  // True if the layer is an input layer.
  bool is_input_;
  // True if the layer is an output layer.
  bool is_output_;

  // Number of neurons.
  std::size_t nneurons_;

  // Cutoff of the input of neuron errors.
  // If errorinput_cutoff_ > 0, then
  // 1) if the input of a neuron error is larger than errorinput_cutoff_, then use errorinput_cutoff_ instead.
  // 2) if the input of a neuron error is smaller than -errorinput_cutoff_, then use -errorinput_cutoff_ instead.
  ErrorType errorinput_cutoff_;

  // The state of the neuron activations.
  // false - in weighted sum state;
  // true - in activation state. 
  bool ac_state_;
  // The state of the errors_.
  // 0 - in weighted sum state;
  // 1 - in error state.
  bool er_state_;

  // Input of the activations.
  std::vector<ActivationInputType> activationinputs_;
  // Input of the errors.
  std::vector<ErrorInputType> errorinputs_;

  // The errors of the neurons.
  std::vector<ErrorType> errors_;
};

} // namespace neuralnet

#endif
