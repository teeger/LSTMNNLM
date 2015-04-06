#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "neuralnet_types.h"
#include "neuralnet_numeric.h"
#include "neuralnet_hash_connection.h"

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

void NeuralNetHashConnection::ForwardPropagate(const NeuralNetSparseLayer &input,
                                               NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);

  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
       it != input_ac.end(); ++it) {
    size_t j;
    size_t offset = it->first;
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0) {
      // needs to call FastUpdateWeightsMinor
      for (j = 0; j + 7 < noutput_; ) {
        update_minor_forward_propagate_atom(j, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 1, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 2, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 3, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 4, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 5, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 6, it->second, offset, output);
        update_minor_forward_propagate_atom(j + 7, it->second, offset, output);
        j += 8;
      }
      for (; j < noutput_; j++) {
        update_minor_forward_propagate_atom(j, it->second, offset, output);
      }
    } else {
      // no need to call FastUpdateWeightsMinor
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 1, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 2, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 3, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 4, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 5, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 6, it->second * weights_[offset++]);
        output.AccumulateInputForActivation(j + 7, it->second * weights_[offset++]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, it->second * weights_[offset++]);
      }
    }
  }
}

void NeuralNetHashConnection::ForwardPropagateForOutput(const NeuralNetSparseLayer &input,
                                                        NeuralNetLayerBase &output,
                                                        size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  double val = 0;
  for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
       it != input_ac.end(); ++it) {
    size_t offset = it->first + idx;
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0) {
      unordered_map<size_t, int>::iterator mi = lastupdate_.find(offset);
      if (mi == lastupdate_.end()) {
        lastupdate_[offset] = 0;
        mi = lastupdate_.find(offset);
      } 
      if (mi->second < num_updates_) {
        FastUpdateWeightsMinor(offset);
      }
    }
    val += it->second * weights_[offset];
  }
  output.AccumulateInputForActivation(idx, val);
}

void NeuralNetHashConnection::AccumulateGradients(const NeuralNetSparseLayer &input,
                                                  const NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);

  const vector<ErrorType> &output_er = output.errors();
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
       it != input_ac.end(); ++it) {
    size_t j;
    size_t offset = it->first;
    for (j = 0; j + 7 < noutput_; ) {
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 1]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 2]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 3]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 4]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 5]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 6]);
      gradients_touched_.push_back(offset++);
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j + 7]);
      gradients_touched_.push_back(offset++);
      j += 8;
    }
    for (; j < noutput_; j++) {
      gradients_[offset] += static_cast<GradientType>(it->second * output_er[j]);
      gradients_touched_.push_back(offset++);
    }
  }
}

void NeuralNetHashConnection::AccumulateGradientsForOutput(const NeuralNetSparseLayer &input,
                                                           const NeuralNetLayerBase &output,
                                                           size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);

  const ErrorType &output_er = output.errors(idx);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
       it != input_ac.end(); ++it) {
    size_t offset = it->first + idx;
    gradients_[offset] += static_cast<GradientType>(it->second * output_er);
    gradients_touched_.push_back(offset);
  }
}

void NeuralNetHashConnection::FastUpdateWeightsMajor(float learning_rate) {
  // FastUpdateWeightsMinor must be called before learning rate changes.
  assert(last_learning_rate_ == -1 || learning_rate == last_learning_rate_);
  last_learning_rate_ = learning_rate;

  // Process the gradients_touched_.
  sort(gradients_touched_.begin(), gradients_touched_.end());
  vector<size_t>::iterator last = unique(gradients_touched_.begin(), gradients_touched_.end());
  gradients_touched_.erase(last, gradients_touched_.end());

  num_updates_++;
  if (!adagrad_) {
    for (vector<size_t>::iterator it = gradients_touched_.begin();
         it != gradients_touched_.end(); ++it) {
      WeightType &wt = weights_[*it];
      GradientType &gt = gradients_[*it];
      const GradientType gt_l2 = gt - l2_regularization_param_ * wt;
      wt += static_cast<WeightType>(learning_rate * gt_l2);

      // Clear gradients.
      gt = 0.0f;

      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      }
    }
  } else {
    for (vector<size_t>::iterator it = gradients_touched_.begin();
         it != gradients_touched_.end(); ++it) {
      WeightType &wt = weights_[*it];
      GradientType &gt = gradients_[*it];
      GradientType &var = sum_gradient_squares_[*it];

      const GradientType gt_l2 = gt - l2_regularization_param_ * wt;
      var += gt_l2 * gt_l2;
      wt += static_cast<WeightType>(learning_rate * gt_l2 / sqrt(var));

      // Clear gradients.
      gt = 0.0f;

      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      }
    }
  }

  // Clear gradients_touched_.
  gradients_touched_.clear();
}

void NeuralNetHashConnection::FastUpdateWeightsMinor() {
  assert(last_learning_rate_ > 0);

  if (l2_regularization_param_ > 0) {
    if (!adagrad_) {
      for (unordered_map<size_t, int>::iterator mi = lastupdate_.begin(); mi != lastupdate_.end(); ++mi) {
        assert(mi->second <= num_updates_);
        assert(mi->second > 0);
        if (mi->second < num_updates_) {
          double scale = pow(1 - last_learning_rate_ * l2_regularization_param_, num_updates_ - mi->second);
          WeightType &wt = weights_[mi->first];
          wt = static_cast<WeightType>(scale * wt);
        }
        mi->second = num_updates_;
      }
    } else {
      for (unordered_map<size_t, int>::iterator mi = lastupdate_.begin(); mi != lastupdate_.end(); ++mi) {
        assert(mi->second <= num_updates_);
        assert(mi->second > 0);
        WeightType &wt = weights_[mi->first];
        GradientType &var = sum_gradient_squares_[mi->first];
        for (int t = mi->second; t < num_updates_; t++) {
          const GradientType gt_l2 = -l2_regularization_param_ * wt;
          var += gt_l2 * gt_l2;
          wt += static_cast<WeightType>(last_learning_rate_ * gt_l2 / sqrt(var));
        }
        mi->second = num_updates_;
      }
    }
  }
  // else, lastupdate_ is unused.

  last_learning_rate_ = -1;
}

void NeuralNetHashConnection::WriteConnectionToTxt(ostream &os) {
  os << "ninput_: " << ninput_ << endl;
  os << "noutput_: " << noutput_ << endl;
  os << "l2_regularization_param_: " << l2_regularization_param_ << endl;
  size_t i = 0;
  for (vector<WeightType>::const_iterator it = weights_.begin();
       it != weights_.end(); ++it, ++i) {
    if (*it == 0) {
      continue;
    }
    os << "weights_[" << i << "]: " << *it << endl;
  }
}

void NeuralNetHashConnection::AllocateConnection() {
  weights_.resize(ninput_ + noutput_);
  fill(weights_.begin(), weights_.end(), 0.0f);

  gradients_.resize(ninput_ + noutput_);
  fill(gradients_.begin(), gradients_.end(), 0.0f);
  
  sum_gradient_squares_.resize(ninput_ + noutput_);
  fill(sum_gradient_squares_.begin(), sum_gradient_squares_.end(), 1.0f);

  lastupdate_.clear();
  gradients_touched_.clear();

  num_updates_ = 0;
  last_learning_rate_ = -1;
}

void NeuralNetHashConnection::FastUpdateWeightsMinor(size_t idx) {
  assert(last_learning_rate_ > 0);
  assert(lastupdate_[idx] < num_updates_);
  assert(l2_regularization_param_ > 0);
  assert(idx < ninput_);

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
