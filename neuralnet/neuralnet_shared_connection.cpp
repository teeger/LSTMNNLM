#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "neuralnet_types.h"
#include "neuralnet_shared_connection.h"

// <cstdio>
using std::size_t;
// <cmath>
using std::pow;
using std::sqrt;
// <iostream>
using std::cerr;
using std::ostream;
using std::endl;
// <vector>
using std::vector;
// <unordered_map>
using std::unordered_map;
// <algorithm>
using std::fill;
using std::sort;
using std::unique;

namespace neuralnet {

void NeuralNetSharedConnection::RandomlyInitialize(boost::mt19937 &rng_engine) {
  if (storage_input_major_) {
    for (size_t i = 0; i < ninput_; i++) {
      // Follow the same design as Mikolov's rnnlm-0.3e codes.
      WeightType r = static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      weights_[i] = r;
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      // Follow the same design as Mikolov's rnnlm-0.3e codes.
      WeightType r = static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
      weights_[j] = r;
    }
  }
}

void NeuralNetSharedConnection::ForwardPropagate(const NeuralNetLayer &input,
                                           NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    size_t i = 0;
    for (vector<ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it, ++i) {
      if (*it == 0) {
        continue;
      }
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[i] < num_updates_) {
        FastUpdateWeightsMinor(i);
      }
      const ActivationInputType val = *it * weights_[i];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, val);
        output.AccumulateInputForActivation(j + 1, val);
        output.AccumulateInputForActivation(j + 2, val);
        output.AccumulateInputForActivation(j + 3, val);
        output.AccumulateInputForActivation(j + 4, val);
        output.AccumulateInputForActivation(j + 5, val);
        output.AccumulateInputForActivation(j + 6, val);
        output.AccumulateInputForActivation(j + 7, val);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, val);
      }
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
        FastUpdateWeightsMinor(j);
      }
      const WeightType wt = weights_[j];
      size_t i;
      double val = 0;
      for (i = 0; i + 7 < ninput_; ) {
        val += input_ac[i] * wt;
        val += input_ac[i + 1] * wt;
        val += input_ac[i + 2] * wt;
        val += input_ac[i + 3] * wt;
        val += input_ac[i + 4] * wt;
        val += input_ac[i + 5] * wt;
        val += input_ac[i + 6] * wt;
        val += input_ac[i + 7] * wt;
        i += 8;
      }
      for (; i < ninput_; i++) {
        val += input_ac[i] * wt;
      }
      output.AccumulateInputForActivation(j, val);
    }
  }
}

void NeuralNetSharedConnection::ForwardPropagate(const NeuralNetSparseLayer &input,
                                           NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[it->first] < num_updates_) {
        FastUpdateWeightsMinor(it->first);
      }
      const ActivationInputType val = it->second * weights_[it->first];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, val);
        output.AccumulateInputForActivation(j + 1, val);
        output.AccumulateInputForActivation(j + 2, val);
        output.AccumulateInputForActivation(j + 3, val);
        output.AccumulateInputForActivation(j + 4, val);
        output.AccumulateInputForActivation(j + 5, val);
        output.AccumulateInputForActivation(j + 6, val);
        output.AccumulateInputForActivation(j + 7, val);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, val);
      }
    }
  } else {
    size_t j;
    for (j = 0; j < noutput_; j++) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
        FastUpdateWeightsMinor(j);
      }
      const WeightType wt = weights_[j];
      double val = 0;
      for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
           it != input_ac.end(); ++it) {
        val += it->second * wt;
      }
      output.AccumulateInputForActivation(j, val);
    }
  }
}

// Note: Passing object by reference is fine, input.activations() calls the derived
// class method. Pass by value will make the cast, but will fail to compile in
// this case since NeuralNetLayerBase is an abstract class.
// Check terminology "slicing".
void NeuralNetSharedConnection::ForwardPropagateForInput(const NeuralNetLayerBase &input,
                                           NeuralNetLayerBase &output,
                                           size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < ninput_);
  const ActivationType input_ac = input.activations(idx);
  if (input_ac != 0) {
    if (storage_input_major_) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[idx] < num_updates_) {
        FastUpdateWeightsMinor(idx);
      }
      const ActivationInputType val = input_ac * weights_[idx];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, val);
        output.AccumulateInputForActivation(j + 1, val);
        output.AccumulateInputForActivation(j + 2, val);
        output.AccumulateInputForActivation(j + 3, val);
        output.AccumulateInputForActivation(j + 4, val);
        output.AccumulateInputForActivation(j + 5, val);
        output.AccumulateInputForActivation(j + 6, val);
        output.AccumulateInputForActivation(j + 7, val);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, val);
      }
    } else {
      cerr << "Warning: ForwardPropagateForInput is not recommended when storage_input_major_ is false!" << endl;
      // Note: if storage_input_major_ is False, recommend to use
      // ForwardPropagateForOutput instead.
      for (size_t j = 0; j < noutput_; j++) {
        if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
          FastUpdateWeightsMinor(j);
        }
        output.AccumulateInputForActivation(j, input_ac * weights_[j]);
      }
    }
  }
}

void NeuralNetSharedConnection::ForwardPropagateForOutput(const NeuralNetLayer &input,
                                           NeuralNetLayerBase &output,
                                           size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    double val = 0;
    for (size_t i = 0; i < ninput_; i++) {
      ActivationType ac = input_ac[i];
      if (ac == 0) {
        continue;
      }
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[i] < num_updates_) {
        FastUpdateWeightsMinor(i);
      }
      val += ac * weights_[i];
    }
    output.AccumulateInputForActivation(idx, val);
  } else {
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[idx] < num_updates_) {
      FastUpdateWeightsMinor(idx);
    }
    const WeightType wt = weights_[idx];
    size_t i;
    double val = 0;
    for (i = 0; i + 7 < ninput_; ) {
      val += input_ac[i] * wt;
      val += input_ac[i + 1] * wt;
      val += input_ac[i + 2] * wt;
      val += input_ac[i + 3] * wt;
      val += input_ac[i + 4] * wt;
      val += input_ac[i + 5] * wt;
      val += input_ac[i + 6] * wt;
      val += input_ac[i + 7] * wt;
      i += 8;
    }
    for (; i < ninput_; i++) {
      val += input_ac[i] * wt;
    }
    output.AccumulateInputForActivation(idx, val);
  }
}

void NeuralNetSharedConnection::ForwardPropagateForOutput(const NeuralNetSparseLayer &input,
                                           NeuralNetLayerBase &output,
                                           size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    double val = 0;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[it->first] < num_updates_) {
        FastUpdateWeightsMinor(it->first);
      }
      val += it->second * weights_[it->first];
    }
    output.AccumulateInputForActivation(idx, val);
  } else {
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[idx] < num_updates_) {
      FastUpdateWeightsMinor(idx);
    }
    const WeightType wt = weights_[idx];
    double val = 0;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      val += it->second * wt;
    }
    output.AccumulateInputForActivation(idx, val);
  }
}

void NeuralNetSharedConnection::BackPropagate(const NeuralNetLayerBase &output,
                                        NeuralNetLayerBase &input) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  const vector<ErrorType> &output_er = output.errors();
  if (storage_input_major_) {
    for (size_t i = 0; i < ninput_; i++) {
      const WeightType wt = weights_[i];
      double val = 0;
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        val += output_er[j] * wt;
        val += output_er[j + 1] * wt;
        val += output_er[j + 2] * wt;
        val += output_er[j + 3] * wt;
        val += output_er[j + 4] * wt;
        val += output_er[j + 5] * wt;
        val += output_er[j + 6] * wt;
        val += output_er[j + 7] * wt;
        j += 8;
      }
      for (; j < noutput_; j++) {
        val += output_er[j] * wt;
      }
      input.AccumulateInputForError(i, val);
    }
  } else {
    size_t j = 0;
    for (vector<ErrorType>::const_iterator it = output_er.begin();
         it != output_er.end(); ++it, ++j) {
      const ErrorInputType val = *it * weights_[j];
      size_t i;
      for (i = 0; i + 7 < ninput_; ) {
        input.AccumulateInputForError(i, val);
        input.AccumulateInputForError(i + 1, val);
        input.AccumulateInputForError(i + 2, val);
        input.AccumulateInputForError(i + 3, val);
        input.AccumulateInputForError(i + 4, val);
        input.AccumulateInputForError(i + 5, val);
        input.AccumulateInputForError(i + 6, val);
        input.AccumulateInputForError(i + 7, val);
        i += 8;
      }
      for (; i < ninput_; i++) {
        input.AccumulateInputForError(i, val);
      }
    }
  }
}

void NeuralNetSharedConnection::BackPropagateForInput(const NeuralNetLayerBase &output,
                                                NeuralNetLayerBase &input,
                                                size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < ninput_);
  const vector<ErrorType> &output_er = output.errors();
  if (storage_input_major_) {
    ErrorType val = 0;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      val += output_er[j];
      val += output_er[j + 1];
      val += output_er[j + 2];
      val += output_er[j + 3];
      val += output_er[j + 4];
      val += output_er[j + 5];
      val += output_er[j + 6];
      val += output_er[j + 7];
      j += 8;
    }
    for (; j < noutput_; j++) {
      val += output_er[j];
    }
    input.AccumulateInputForError(idx, weights_[idx] * val);
  } else {
    cerr << "Warning: BackPropagateForInput is not recommended when storage_input_major_ is false!" << endl;
    double val = 0;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      val += output_er[j] * weights_[j];
      val += output_er[j + 1] * weights_[j + 1];
      val += output_er[j + 2] * weights_[j + 2];
      val += output_er[j + 3] * weights_[j + 3];
      val += output_er[j + 4] * weights_[j + 4];
      val += output_er[j + 5] * weights_[j + 5];
      val += output_er[j + 6] * weights_[j + 6];
      val += output_er[j + 7] * weights_[j + 7];
    }
    for (; j < noutput_; j++) {
      val += output_er[j] * weights_[j];
    }
    input.AccumulateInputForError(idx, val);
  }
}

void NeuralNetSharedConnection::BackPropagateForOutput(const NeuralNetLayerBase &output,
                                                 NeuralNetLayerBase &input,
                                                 size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const ErrorType output_er = output.errors(idx);
  if (storage_input_major_) {
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      input.AccumulateInputForError(i, output_er * weights_[i]);
      input.AccumulateInputForError(i + 1, output_er * weights_[i + 1]);
      input.AccumulateInputForError(i + 2, output_er * weights_[i + 2]);
      input.AccumulateInputForError(i + 3, output_er * weights_[i + 3]);
      input.AccumulateInputForError(i + 4, output_er * weights_[i + 4]);
      input.AccumulateInputForError(i + 5, output_er * weights_[i + 5]);
      input.AccumulateInputForError(i + 6, output_er * weights_[i + 6]);
      input.AccumulateInputForError(i + 7, output_er * weights_[i + 7]);
    }
    for (; i < ninput_; i++) {
      input.AccumulateInputForError(i, output_er * weights_[i]);
    }
  } else {
    const ErrorInputType val = output_er * weights_[idx];
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      input.AccumulateInputForError(i, val);
      input.AccumulateInputForError(i + 1, val);
      input.AccumulateInputForError(i + 2, val);
      input.AccumulateInputForError(i + 3, val);
      input.AccumulateInputForError(i + 4, val);
      input.AccumulateInputForError(i + 5, val);
      input.AccumulateInputForError(i + 6, val);
      input.AccumulateInputForError(i + 7, val);
      i += 8;
    }
    for (; i < ninput_; i++) {
      input.AccumulateInputForError(i, val);
    }
  }
}

void NeuralNetSharedConnection::AccumulateGradients(const NeuralNetLayer &input,
                                              const NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);

  const vector<ErrorType> &output_er = output.errors();
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    size_t i = 0;
    for (vector<ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it, ++i) {
      if (*it == 0) {
        continue;
      }
      ErrorType val = 0;
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        val += output_er[j];
        val += output_er[j + 1];
        val += output_er[j + 2];
        val += output_er[j + 3];
        val += output_er[j + 4];
        val += output_er[j + 5];
        val += output_er[j + 6];
        val += output_er[j + 7];
        j += 8;
      }
      for (; j < noutput_; j++) {
        val += output_er[j];
      }
      gradients_[i] += static_cast<GradientType>(*it * val);
      gradients_touched_.push_back(i);
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      ErrorType er = output_er[j];
      ActivationInputType val = 0;
      size_t i;
      for (i = 0; i + 7 < ninput_; ) {
        val += input_ac[i];
        val += input_ac[i + 1];
        val += input_ac[i + 2];
        val += input_ac[i + 3];
        val += input_ac[i + 4];
        val += input_ac[i + 5];
        val += input_ac[i + 6];
        val += input_ac[i + 7];
        i += 8;
      }
      for (; i < ninput_; i++) {
        val += input_ac[i];
      }
      gradients_[j] += static_cast<GradientType>(er * val);
      gradients_touched_.push_back(j);
    }
  }
}

void NeuralNetSharedConnection::AccumulateGradients(const NeuralNetSparseLayer &input,
                                              const NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);

  const vector<ErrorType> &output_er = output.errors();
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      ErrorType val = 0;
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        val += output_er[j];
        val += output_er[j + 1];
        val += output_er[j + 2];
        val += output_er[j + 3];
        val += output_er[j + 4];
        val += output_er[j + 5];
        val += output_er[j + 6];
        val += output_er[j + 7];
        j += 8;
      }
      for (; j < noutput_; j++) {
        val += output_er[j];
      }
      gradients_[it->first] += static_cast<GradientType>(it->second * val);
      gradients_touched_.push_back(it->first);
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      ErrorType er = output_er[j];
      ActivationType val = 0;
      for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
           it != input_ac.end(); ++it) {
        val += it->second;
      }
      gradients_[j] = static_cast<GradientType>(er * val);
      gradients_touched_.push_back(j);
    }
  }
}

void NeuralNetSharedConnection::AccumulateGradientsForInput(const NeuralNetLayerBase &input,
                                                      const NeuralNetLayerBase &output,
                                                      size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < ninput_);

  const vector<ErrorType> &output_er = output.errors();
  const ActivationType input_ac = input.activations(idx);
  if (storage_input_major_) {
    if (input_ac != 0) {
      ErrorType val = 0;
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        val += output_er[j];
        val += output_er[j + 1];
        val += output_er[j + 2];
        val += output_er[j + 3];
        val += output_er[j + 4];
        val += output_er[j + 5];
        val += output_er[j + 6];
        val += output_er[j + 7];
        j += 8;
      }
      for (; j < noutput_; j++) {
        val += output_er[j];
      }
      gradients_[idx] += static_cast<GradientType>(input_ac * val);
      gradients_touched_.push_back(idx);
    }
  } else {
    cerr << "Warning: AccumulateGradientsForInput is not recommended when storage_input_major_ is false!" << endl;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      gradients_[j] += static_cast<GradientType>(input_ac * output_er[j]);
      gradients_[j + 1] += static_cast<GradientType>(input_ac * output_er[j + 1]);
      gradients_[j + 2] += static_cast<GradientType>(input_ac * output_er[j + 2]);
      gradients_[j + 3] += static_cast<GradientType>(input_ac * output_er[j + 3]);
      gradients_[j + 4] += static_cast<GradientType>(input_ac * output_er[j + 4]);
      gradients_[j + 5] += static_cast<GradientType>(input_ac * output_er[j + 5]);
      gradients_[j + 6] += static_cast<GradientType>(input_ac * output_er[j + 6]);
      gradients_[j + 7] += static_cast<GradientType>(input_ac * output_er[j + 7]);

      gradients_touched_.push_back(j);
      gradients_touched_.push_back(j + 1);
      gradients_touched_.push_back(j + 2);
      gradients_touched_.push_back(j + 3);
      gradients_touched_.push_back(j + 4);
      gradients_touched_.push_back(j + 5);
      gradients_touched_.push_back(j + 6);
      gradients_touched_.push_back(j + 7);
    }
    for (; j < noutput_; j++) {
      gradients_[j] += static_cast<GradientType>(input_ac * output_er[j]);
    }
  }
}

void NeuralNetSharedConnection::AccumulateGradientsForOutput(const NeuralNetLayer &input,
                                                       const NeuralNetLayerBase &output,
                                                       size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);

  const ErrorType output_er = output.errors(idx);
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    size_t i = 0;
    for (vector<ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it, ++i) {
      if (*it == 0) {
        continue;
      }
      gradients_[i] += static_cast<GradientType>(*it * output_er);
      gradients_touched_.push_back(i);
    }
  } else {
    ActivationType val = 0;
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      val += input_ac[i];
      val += input_ac[i + 1];
      val += input_ac[i + 2];
      val += input_ac[i + 3];
      val += input_ac[i + 4];
      val += input_ac[i + 5];
      val += input_ac[i + 6];
      val += input_ac[i + 7];

      i += 8;
    }
    for (; i < ninput_; i++) {
      val += input_ac[i];
    }
    gradients_[idx] += static_cast<GradientType>(output_er * val);
    gradients_touched_.push_back(idx);
  }
}

void NeuralNetSharedConnection::AccumulateGradientsForOutput(const NeuralNetSparseLayer &input,
                                                       const NeuralNetLayerBase &output,
                                                       size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);

  const ErrorType &output_er = output.errors(idx);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      gradients_[it->first] += static_cast<GradientType>(it->second * output_er);
      gradients_touched_.push_back(it->first);
    }
  } else {
    ActivationType val = 0;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      val += it->second;
    }
    gradients_[idx] = static_cast<GradientType>(output_er * val);
    gradients_touched_.push_back(idx);
  }
}

void NeuralNetSharedConnection::FastUpdateWeightsMajor(float learning_rate) {
  // FastUpdateWeightsMinor must be called before learning rate changes.
  assert(last_learning_rate_ == -1 || learning_rate == last_learning_rate_);
  last_learning_rate_ = learning_rate;

  // Process the gradients_touched_.
  sort(gradients_touched_.begin(), gradients_touched_.end());
  vector<size_t>::iterator last = unique(gradients_touched_.begin(), gradients_touched_.end());
  gradients_touched_.erase(last, gradients_touched_.end());

  num_updates_++;
  size_t dim;
  if (storage_input_major_) {
    dim = noutput_;
  } else {
    dim = ninput_;
  }
  if (!adagrad_) {
    for (vector<size_t>::iterator it = gradients_touched_.begin();
         it != gradients_touched_.end(); ++it) {
      GradientType &gt = gradients_[*it];
      weights_[*it] += static_cast<WeightType>(
          learning_rate * (gt - l2_regularization_param_ * weights_[*it]));
      
      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      }

      // Clear gradients.
      gt = 0.0f;
    }
  } else {
    for (vector<size_t>::iterator it = gradients_touched_.begin();
      it != gradients_touched_.end(); ++it) {
      GradientType &gt = gradients_[*it];
      const GradientType gt_l2 = gt - l2_regularization_param_ * weights_[*it];
      sum_gradient_squares_[*it] += gt_l2 * gt_l2;
      weights_[*it] += static_cast<WeightType>(learning_rate * gt_l2 / sqrt(sum_gradient_squares_[*it]));

      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      } 

      // Clear gradients.
      gt = 0.0f;
    }
  }

  // Clear gradients_touched_.
  gradients_touched_.clear();
}

void NeuralNetSharedConnection::FastUpdateWeightsMinor() {
  assert(last_learning_rate_ > 0);

  if (l2_regularization_param_ > 0) {
    size_t dim1;
    if (storage_input_major_) {
      dim1 = ninput_;
    }
    else {
      dim1 = noutput_;
    }

    if (!adagrad_) {
      for (size_t d1 = 0; d1 < dim1; d1++) {
        int &lu = lastupdate_[d1];
        if (lu < num_updates_) {
          double scale = pow(1 - last_learning_rate_ * l2_regularization_param_, num_updates_ - lastupdate_[d1]);
          weights_[d1] = static_cast<WeightType>(scale * weights_[d1]);
        }
        lu = num_updates_;
      }
    } else {
      for (size_t d1 = 0; d1 < dim1; d1++) {
        int &lu = lastupdate_[d1];
        WeightType &wt = weights_[d1];
        GradientType &var = sum_gradient_squares_[d1];
        for (int t = lu; t < num_updates_; t++) {
          const GradientType gt_l2 = -l2_regularization_param_ * wt;
          var += gt_l2 * gt_l2;
          wt += static_cast<WeightType>(last_learning_rate_ * gt_l2 / sqrt(var));
        }
        lu = num_updates_;
      }
    }
  }
  // else, lastupdate_ is unused.

  last_learning_rate_ = -1;
}

void NeuralNetSharedConnection::WriteConnectionToTxt(ostream &os) {
  os << "storage_row_major: " << storage_input_major_ << endl;
  os << "ninput_: " << ninput_ << endl;
  os << "noutput_: " << noutput_ << endl;
  os << "l2_regularization_param_: " << l2_regularization_param_ << endl;
  size_t dim1;
  if (storage_input_major_) {
    dim1 = ninput_;
  } else {
    dim1 = noutput_;
  }
  for (size_t d1 = 0; d1 < dim1; d1++) {
    os << "weights_[" << d1 << "]: " << weights_[d1] << endl;
  }
}

void NeuralNetSharedConnection::AllocateConnection() {
  size_t dim1;
  if (storage_input_major_) {
    dim1 = ninput_;
  } else {
    dim1 = noutput_;
  }
    
  weights_.resize(dim1);
  fill(weights_.begin(), weights_.end(), 0.0f);

  gradients_.resize(dim1);
  fill(gradients_.begin(), gradients_.end(), 0.0f);
  
  sum_gradient_squares_.resize(dim1);
  // Note: Empirically, it is good to initialize it as 1.0f. If something
  // weired happens, come here and check the value.
  // The idea is to make sure the approximated Hessian is invertible (check
  // original paper).
  fill(sum_gradient_squares_.begin(), sum_gradient_squares_.end(), 1.0f);

  lastupdate_.resize(dim1);
  fill(lastupdate_.begin(), lastupdate_.end(), 0);

  gradients_touched_.clear();

  num_updates_ = 0;
  last_learning_rate_ = -1;
}

void NeuralNetSharedConnection::FastUpdateWeightsMinor(size_t idx) {
  assert(last_learning_rate_ > 0);
  assert(lastupdate_[idx] < num_updates_);
  assert(l2_regularization_param_ > 0);
  assert((idx < ninput_ && storage_input_major_) || (idx < noutput_ && !storage_input_major_));

  WeightType &wt = weights_[idx];
  if (wt != 0) {
    if (!adagrad_) {
      double scale = pow(1 - last_learning_rate_ * l2_regularization_param_, num_updates_ - lastupdate_[idx]);
      wt = static_cast<WeightType>(scale * wt);
    } else {
      GradientType &var = sum_gradient_squares_[idx];
      for (int t = lastupdate_[idx]; t < num_updates_; t++) {
        const GradientType gt_l2 = -l2_regularization_param_ * wt;
        var += gt_l2 * gt_l2;
        wt += static_cast<WeightType>(last_learning_rate_ * gt_l2 / sqrt(var));
      }
    }
  }
  lastupdate_[idx] = num_updates_;
}

} // namespace neuralnet
