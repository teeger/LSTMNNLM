#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "neuralnet_types.h"
#include "neuralnet_connection.h"

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

void NeuralNetConnection::RandomlyInitialize(boost::mt19937 &rng_engine) {
  if (storage_input_major_) {
    for (size_t i = 0; i < ninput_; i++) {
      vector<WeightType> &wt = weights_[i];
      for (size_t j = 0; j < noutput_; j++) {
        // Follow the same design as Mikolov's rnnlm-0.3e codes.
        WeightType r = static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        wt[j] = r;
      }
    }
  } else {
    // Note: Temporarily keep the same order as storage_row_maror_ == True, to get consistent results.
    for (size_t i = 0; i < ninput_; i++) {
      for (size_t j = 0; j < noutput_; j++) {
        // Follow the same design as Mikolov's rnnlm-0.3e codes.
        WeightType r = static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
        weights_[j][i] = r;
      }
    }
    /*
       for (size_t j = 0; j < noutput_; j++) {
       vector<WeightType> &wt = weights_[j];
       for (size_t i = 0; i < ninput_; i++) {
// Follow the same design as Mikolov's rnnlm-0.3e codes.
WeightType r = static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
r += static_cast<WeightType>(gen_random_double_01(rng_engine)*0.2 - 0.1);
wt[i] = r;
}
}
*/
}
}

/*
   void NeuralNetConnection::L2NormalizeWeights(bool by_row) {
   size_t dim1, dim2;
   if (storage_input_major_) {
   dim1 = ninput_;
   dim2 = noutput_;
   } else {
   dim1 = noutput_;
   dim2 = ninput_;
   }

   if (storage_input_major_ == by_row) {
   for (size_t d1 = 0; d1 < dim1; d1++) {
   WeightType total = 0;
   vector<WeightType> &wt = weights_[d1];
   for (size_t d2 = 0; d2 < dim2; d2++) {
   total += wt[d2] * wt[d2];
   }
   if (total == 0) {
   continue;
   }
   total = sqrt(total);
   for (size_t d2 = 0; d2 < dim2; d2++) {
   wt[d2] /= total;
   }
   }
   } else {
   for (size_t d2 = 0; d2 < dim2; d2++) {
   WeightType total = 0;
   for (size_t d1 = 0; d1 < dim1; d1++) {
   total += weights_[d1][d2] * weights_[d1][d2];
   }
   if (total == 0) {
   continue;
   }
   total = sqrt(total);
   for (size_t d1 = 0; d1 < dim1; d1++) {
   weights_[d1][d2] /= total;
   }
   }
   }
   }
   */

void NeuralNetConnection::ForwardPropagate(const NeuralNetLayer &input,
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
      const vector<WeightType> &wt = weights_[i];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, *it * wt[j]);
        output.AccumulateInputForActivation(j + 1, *it * wt[j + 1]);
        output.AccumulateInputForActivation(j + 2, *it * wt[j + 2]);
        output.AccumulateInputForActivation(j + 3, *it * wt[j + 3]);
        output.AccumulateInputForActivation(j + 4, *it * wt[j + 4]);
        output.AccumulateInputForActivation(j + 5, *it * wt[j + 5]);
        output.AccumulateInputForActivation(j + 6, *it * wt[j + 6]);
        output.AccumulateInputForActivation(j + 7, *it * wt[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, *it * wt[j]);
      }
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
        FastUpdateWeightsMinor(j);
      }
      const vector<WeightType> &wt = weights_[j];
      size_t i;
      double val = 0;
      for (i = 0; i + 7 < ninput_; ) {
        val += input_ac[i] * wt[i];
        val += input_ac[i + 1] * wt[i + 1];
        val += input_ac[i + 2] * wt[i + 2];
        val += input_ac[i + 3] * wt[i + 3];
        val += input_ac[i + 4] * wt[i + 4];
        val += input_ac[i + 5] * wt[i + 5];
        val += input_ac[i + 6] * wt[i + 6];
        val += input_ac[i + 7] * wt[i + 7];
        i += 8;
      }
      for (; i < ninput_; i++) {
        val += input_ac[i] * wt[i];
      }
      output.AccumulateInputForActivation(j, val);
    }
  }
}

void NeuralNetConnection::ForwardPropagate(const NeuralNetSparseLayer &input,
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
      const vector<WeightType> &wt = weights_[it->first];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, it->second * wt[j]);
        output.AccumulateInputForActivation(j + 1, it->second * wt[j + 1]);
        output.AccumulateInputForActivation(j + 2, it->second * wt[j + 2]);
        output.AccumulateInputForActivation(j + 3, it->second * wt[j + 3]);
        output.AccumulateInputForActivation(j + 4, it->second * wt[j + 4]);
        output.AccumulateInputForActivation(j + 5, it->second * wt[j + 5]);
        output.AccumulateInputForActivation(j + 6, it->second * wt[j + 6]);
        output.AccumulateInputForActivation(j + 7, it->second * wt[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, it->second * wt[j]);
      }
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
        FastUpdateWeightsMinor(j);
      }
      const vector<WeightType> &wt = weights_[j];
      double val = 0;
      for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
           it != input_ac.end(); ++it) {
        val += it->second * wt[it->first];
      }
      output.AccumulateInputForActivation(j, val);
    }
  }
}

// Note: Passing object by reference is fine, input.activations() calls the derived
// class method. Pass by value will make the cast, but will fail to compile in
// this case since NeuralNetLayerBase is an abstract class.
// Check terminology "slicing".
void NeuralNetConnection::ForwardPropagateForInput(const NeuralNetLayerBase &input,
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
      const vector<WeightType> &wt = weights_[idx];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        output.AccumulateInputForActivation(j, input_ac * wt[j]);
        output.AccumulateInputForActivation(j + 1, input_ac * wt[j + 1]);
        output.AccumulateInputForActivation(j + 2, input_ac * wt[j + 2]);
        output.AccumulateInputForActivation(j + 3, input_ac * wt[j + 3]);
        output.AccumulateInputForActivation(j + 4, input_ac * wt[j + 4]);
        output.AccumulateInputForActivation(j + 5, input_ac * wt[j + 5]);
        output.AccumulateInputForActivation(j + 6, input_ac * wt[j + 6]);
        output.AccumulateInputForActivation(j + 7, input_ac * wt[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        output.AccumulateInputForActivation(j, input_ac * wt[j]);
      }
    } else {
      cerr << "Warning: ForwardPropagateForInput is not recommended when storage_input_major_ is false!" << endl;
      // Note: if storage_input_major_ is False, recommend to use
      // ForwardPropagateForOutput instead.
      for (size_t j = 0; j < noutput_; j++) {
        if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[j] < num_updates_) {
          FastUpdateWeightsMinor(j);
        }
        output.AccumulateInputForActivation(j, input_ac * weights_[j][idx]);
      }
    }
  }
}

void NeuralNetConnection::ForwardPropagateForOutput(const NeuralNetLayer &input,
                                                    NeuralNetLayerBase &output,
                                                    size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    cerr << "Warning: ForwardPropagateForOutput is not recommended when storage_input_major_ is true!" << endl;
    double val = 0;
    for (size_t i = 0; i < ninput_; i++) {
      ActivationType ac = input_ac[i];
      if (ac == 0) {
        continue;
      }
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[i] < num_updates_) {
        FastUpdateWeightsMinor(i);
      }
      val += ac * weights_[i][idx];
    }
    output.AccumulateInputForActivation(idx, val);
  } else {
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[idx] < num_updates_) {
      FastUpdateWeightsMinor(idx);
    }
    const vector<WeightType> &wt = weights_[idx];
    size_t i;
    double val = 0;
    for (i = 0; i + 7 < ninput_; ) {
      val += input_ac[i] * wt[i];
      val += input_ac[i + 1] * wt[i + 1];
      val += input_ac[i + 2] * wt[i + 2];
      val += input_ac[i + 3] * wt[i + 3];
      val += input_ac[i + 4] * wt[i + 4];
      val += input_ac[i + 5] * wt[i + 5];
      val += input_ac[i + 6] * wt[i + 6];
      val += input_ac[i + 7] * wt[i + 7];
      i += 8;
    }
    for (; i < ninput_; i++) {
      val += input_ac[i] * wt[i];
    }
    output.AccumulateInputForActivation(idx, val);
  }
}

void NeuralNetConnection::ForwardPropagateForOutput(const NeuralNetSparseLayer &input,
                                                    NeuralNetLayerBase &output,
                                                    size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    cerr << "Warning: ForwardPropagateForOutput is not recommended when storage_input_major_ is true!" << endl;
    double val = 0;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[it->first] < num_updates_) {
        FastUpdateWeightsMinor(it->first);
      }
      val += it->second * weights_[it->first][idx];
    }
    output.AccumulateInputForActivation(idx, val);
  } else {
    if (last_learning_rate_ > 0 && l2_regularization_param_ > 0 && lastupdate_[idx] < num_updates_) {
      FastUpdateWeightsMinor(idx);
    }
    const vector<WeightType> &wt = weights_[idx];
    double val = 0;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      val += it->second * wt[it->first];
    }
    output.AccumulateInputForActivation(idx, val);
  }
}

void NeuralNetConnection::BackPropagate(const NeuralNetLayerBase &output,
                                        NeuralNetLayerBase &input) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  const vector<ErrorType> &output_er = output.errors();
  if (storage_input_major_) {
    for (size_t i = 0; i < ninput_; i++) {
      const vector<WeightType> &wt = weights_[i];
      double val = 0;
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        val += output_er[j] * wt[j];
        val += output_er[j + 1] * wt[j + 1];
        val += output_er[j + 2] * wt[j + 2];
        val += output_er[j + 3] * wt[j + 3];
        val += output_er[j + 4] * wt[j + 4];
        val += output_er[j + 5] * wt[j + 5];
        val += output_er[j + 6] * wt[j + 6];
        val += output_er[j + 7] * wt[j + 7];
        j += 8;
      }
      for (; j < noutput_; j++) {
        val += output_er[j] * wt[j];
      }
      input.AccumulateInputForError(i, val);
    }
  } else {
    size_t j = 0;
    for (vector<ErrorType>::const_iterator it = output_er.begin();
         it != output_er.end(); ++it, ++j) {
      const vector<WeightType> &wt = weights_[j];
      size_t i;
      for (i = 0; i + 7 < ninput_; ) {
        input.AccumulateInputForError(i, *it * wt[i]);
        input.AccumulateInputForError(i + 1, *it * wt[i + 1]);
        input.AccumulateInputForError(i + 2, *it * wt[i + 2]);
        input.AccumulateInputForError(i + 3, *it * wt[i + 3]);
        input.AccumulateInputForError(i + 4, *it * wt[i + 4]);
        input.AccumulateInputForError(i + 5, *it * wt[i + 5]);
        input.AccumulateInputForError(i + 6, *it * wt[i + 6]);
        input.AccumulateInputForError(i + 7, *it * wt[i + 7]);
        i += 8;
      }
      for (; i < ninput_; i++) {
        input.AccumulateInputForError(i, *it * wt[i]);
      }
    }
  }
}

void NeuralNetConnection::BackPropagateForInput(const NeuralNetLayerBase &output,
                                                NeuralNetLayerBase &input,
                                                size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < ninput_);
  const vector<ErrorType> &output_er = output.errors();
  if (storage_input_major_) {
    const vector<WeightType> &wt = weights_[idx];
    double val = 0;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      val += output_er[j] * wt[j];
      val += output_er[j + 1] * wt[j + 1];
      val += output_er[j + 2] * wt[j + 2];
      val += output_er[j + 3] * wt[j + 3];
      val += output_er[j + 4] * wt[j + 4];
      val += output_er[j + 5] * wt[j + 5];
      val += output_er[j + 6] * wt[j + 6];
      val += output_er[j + 7] * wt[j + 7];
      j += 8;
    }
    for (; j < noutput_; j++) {
      val += output_er[j] * wt[j];
    }
    input.AccumulateInputForError(idx, val);
  } else {
    cerr << "Warning: BackPropagateForInput is not recommended when storage_input_major_ is false!" << endl;
    double val = 0;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      val += output_er[j] * weights_[j][idx];
      val += output_er[j + 1] * weights_[j + 1][idx];
      val += output_er[j + 2] * weights_[j + 2][idx];
      val += output_er[j + 3] * weights_[j + 3][idx];
      val += output_er[j + 4] * weights_[j + 4][idx];
      val += output_er[j + 5] * weights_[j + 5][idx];
      val += output_er[j + 6] * weights_[j + 6][idx];
      val += output_er[j + 7] * weights_[j + 7][idx];
    }
    for (; j < noutput_; j++) {
      val += output_er[j] * weights_[j][idx];
    }
    input.AccumulateInputForError(idx, val);
  }
}

void NeuralNetConnection::BackPropagateForOutput(const NeuralNetLayerBase &output,
                                                 NeuralNetLayerBase &input,
                                                 size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);
  const ErrorType output_er = output.errors(idx);
  if (storage_input_major_) {
    cerr << "Warning: BackPropagateForOutput is not recommended when storage_input_major_ is true!" << endl;
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      input.AccumulateInputForError(i, output_er * weights_[i][idx]);
      input.AccumulateInputForError(i + 1, output_er * weights_[i + 1][idx]);
      input.AccumulateInputForError(i + 2, output_er * weights_[i + 2][idx]);
      input.AccumulateInputForError(i + 3, output_er * weights_[i + 3][idx]);
      input.AccumulateInputForError(i + 4, output_er * weights_[i + 4][idx]);
      input.AccumulateInputForError(i + 5, output_er * weights_[i + 5][idx]);
      input.AccumulateInputForError(i + 6, output_er * weights_[i + 6][idx]);
      input.AccumulateInputForError(i + 7, output_er * weights_[i + 7][idx]);
    }
    for (; i < ninput_; i++) {
      input.AccumulateInputForError(i, output_er * weights_[i][idx]);
    }
  } else {
    const vector<WeightType> &wt = weights_[idx];
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      input.AccumulateInputForError(i, output_er * wt[i]);
      input.AccumulateInputForError(i + 1, output_er * wt[i + 1]);
      input.AccumulateInputForError(i + 2, output_er * wt[i + 2]);
      input.AccumulateInputForError(i + 3, output_er * wt[i + 3]);
      input.AccumulateInputForError(i + 4, output_er * wt[i + 4]);
      input.AccumulateInputForError(i + 5, output_er * wt[i + 5]);
      input.AccumulateInputForError(i + 6, output_er * wt[i + 6]);
      input.AccumulateInputForError(i + 7, output_er * wt[i + 7]);
      i += 8;
    }
    for (; i < ninput_; i++) {
      input.AccumulateInputForError(i, output_er * wt[i]);
    }
  }
}

void NeuralNetConnection::AccumulateGradients(const NeuralNetLayer &input,
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
      vector<GradientType> &gt = gradients_[i];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        gt[j] += static_cast<GradientType>(*it * output_er[j]);
        gt[j + 1] += static_cast<GradientType>(*it * output_er[j + 1]);
        gt[j + 2] += static_cast<GradientType>(*it * output_er[j + 2]);
        gt[j + 3] += static_cast<GradientType>(*it * output_er[j + 3]);
        gt[j + 4] += static_cast<GradientType>(*it * output_er[j + 4]);
        gt[j + 5] += static_cast<GradientType>(*it * output_er[j + 5]);
        gt[j + 6] += static_cast<GradientType>(*it * output_er[j + 6]);
        gt[j + 7] += static_cast<GradientType>(*it * output_er[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        gt[j] += static_cast<GradientType>(*it * output_er[j]);
      }
      gradients_touched_.push_back(i);
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      vector<GradientType> &gt = gradients_[j];
      ErrorType er = output_er[j];
      size_t i;
      for (i = 0; i + 7 < ninput_; ) {
        gt[i] += static_cast<GradientType>(input_ac[i] * er);
        gt[i + 1] += static_cast<GradientType>(input_ac[i + 1] * er);
        gt[i + 2] += static_cast<GradientType>(input_ac[i + 2] * er);
        gt[i + 3] += static_cast<GradientType>(input_ac[i + 3] * er);
        gt[i + 4] += static_cast<GradientType>(input_ac[i + 4] * er);
        gt[i + 5] += static_cast<GradientType>(input_ac[i + 5] * er);
        gt[i + 6] += static_cast<GradientType>(input_ac[i + 6] * er);
        gt[i + 7] += static_cast<GradientType>(input_ac[i + 7] * er);
        i += 8;
      }
      for (; i < ninput_; i++) {
        gt[i] += static_cast<GradientType>(input_ac[i] * er);
      }
      gradients_touched_.push_back(j);
    }
  }
}

void NeuralNetConnection::AccumulateGradients(const NeuralNetSparseLayer &input,
                                              const NeuralNetLayerBase &output) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);

  const vector<ErrorType> &output_er = output.errors();
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      vector<GradientType> &gt = gradients_[it->first];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        gt[j] += static_cast<GradientType>(it->second * output_er[j]);
        gt[j + 1] += static_cast<GradientType>(it->second * output_er[j + 1]);
        gt[j + 2] += static_cast<GradientType>(it->second * output_er[j + 2]);
        gt[j + 3] += static_cast<GradientType>(it->second * output_er[j + 3]);
        gt[j + 4] += static_cast<GradientType>(it->second * output_er[j + 4]);
        gt[j + 5] += static_cast<GradientType>(it->second * output_er[j + 5]);
        gt[j + 6] += static_cast<GradientType>(it->second * output_er[j + 6]);
        gt[j + 7] += static_cast<GradientType>(it->second * output_er[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        gt[j] += static_cast<GradientType>(it->second * output_er[j]);
      }
      gradients_touched_.push_back(it->first);
    }
  } else {
    for (size_t j = 0; j < noutput_; j++) {
      vector<GradientType> &gt = gradients_[j];
      ErrorType er = output_er[j];
      for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
           it != input_ac.end(); ++it) {
        gt[it->first] += static_cast<GradientType>(it->second * er);
      }
      gradients_touched_.push_back(j);
    }
  }
}

void NeuralNetConnection::AccumulateGradientsForInput(const NeuralNetLayerBase &input,
                                                      const NeuralNetLayerBase &output,
                                                      size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < ninput_);

  const vector<ErrorType> &output_er = output.errors();
  const ActivationType input_ac = input.activations(idx);
  if (storage_input_major_) {
    if (input_ac != 0) {
      vector<GradientType> &gt = gradients_[idx];
      size_t j;
      for (j = 0; j + 7 < noutput_; ) {
        gt[j] += static_cast<GradientType>(input_ac * output_er[j]);
        gt[j + 1] += static_cast<GradientType>(input_ac * output_er[j + 1]);
        gt[j + 2] += static_cast<GradientType>(input_ac * output_er[j + 2]);
        gt[j + 3] += static_cast<GradientType>(input_ac * output_er[j + 3]);
        gt[j + 4] += static_cast<GradientType>(input_ac * output_er[j + 4]);
        gt[j + 5] += static_cast<GradientType>(input_ac * output_er[j + 5]);
        gt[j + 6] += static_cast<GradientType>(input_ac * output_er[j + 6]);
        gt[j + 7] += static_cast<GradientType>(input_ac * output_er[j + 7]);
        j += 8;
      }
      for (; j < noutput_; j++) {
        gt[j] += static_cast<GradientType>(input_ac * output_er[j]);
      }
      gradients_touched_.push_back(idx);
    }
  } else {
    cerr << "Warning: AccumulateGradientsForInput is not recommended when storage_input_major_ is false!" << endl;
    size_t j;
    for (j = 0; j + 7 < noutput_; ) {
      gradients_[j][idx] += static_cast<GradientType>(input_ac * output_er[j]);
      gradients_[j + 1][idx] += static_cast<GradientType>(input_ac * output_er[j + 1]);
      gradients_[j + 2][idx] += static_cast<GradientType>(input_ac * output_er[j + 2]);
      gradients_[j + 3][idx] += static_cast<GradientType>(input_ac * output_er[j + 3]);
      gradients_[j + 4][idx] += static_cast<GradientType>(input_ac * output_er[j + 4]);
      gradients_[j + 5][idx] += static_cast<GradientType>(input_ac * output_er[j + 5]);
      gradients_[j + 6][idx] += static_cast<GradientType>(input_ac * output_er[j + 6]);
      gradients_[j + 7][idx] += static_cast<GradientType>(input_ac * output_er[j + 7]);

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
      gradients_[j][idx] += static_cast<GradientType>(input_ac * output_er[j]);
    }
  }
}

void NeuralNetConnection::AccumulateGradientsForOutput(const NeuralNetLayer &input,
                                                       const NeuralNetLayerBase &output,
                                                       size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);

  const ErrorType output_er = output.errors(idx);
  const vector<ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    cerr << "Warning: AccumulateGradientsForOutput is not recommended when storage_input_major_ is true!" << endl;
    size_t i = 0;
    for (vector<ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it, ++i) {
      if (*it == 0) {
        continue;
      }
      gradients_[i][idx] += static_cast<GradientType>(*it * output_er);
      gradients_touched_.push_back(i);
    }
  } else {
    vector<GradientType> &gt = gradients_[idx];
    size_t i;
    for (i = 0; i + 7 < ninput_; ) {
      gt[i] += static_cast<GradientType>(input_ac[i] * output_er);
      gt[i + 1] += static_cast<GradientType>(input_ac[i + 1] * output_er);
      gt[i + 2] += static_cast<GradientType>(input_ac[i + 2] * output_er);
      gt[i + 3] += static_cast<GradientType>(input_ac[i + 3] * output_er);
      gt[i + 4] += static_cast<GradientType>(input_ac[i + 4] * output_er);
      gt[i + 5] += static_cast<GradientType>(input_ac[i + 5] * output_er);
      gt[i + 6] += static_cast<GradientType>(input_ac[i + 6] * output_er);
      gt[i + 7] += static_cast<GradientType>(input_ac[i + 7] * output_er);
      i += 8;
    }
    for (; i < ninput_; i++) {
      gt[i] += static_cast<GradientType>(input_ac[i] * output_er);
    }
    gradients_touched_.push_back(idx);
  }
}

void NeuralNetConnection::AccumulateGradientsForOutput(const NeuralNetSparseLayer &input,
                                                       const NeuralNetLayerBase &output,
                                                       size_t idx) {
  assert(input.nneurons() == ninput_);
  assert(output.nneurons() == noutput_);
  assert(idx < noutput_);

  const ErrorType &output_er = output.errors(idx);
  const unordered_map<size_t, ActivationType> &input_ac = input.activations();
  if (storage_input_major_) {
    cerr << "Warning: AccumulateGradientsForOutput is not recommended when storage_input_major_ is true!" << endl;
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      gradients_[it->first][idx] += static_cast<GradientType>(it->second * output_er);
      gradients_touched_.push_back(it->first);
    }
  } else {
    vector<GradientType> &gt = gradients_[idx];
    for (unordered_map<size_t, ActivationType>::const_iterator it = input_ac.begin();
         it != input_ac.end(); ++it) {
      gt[it->first] += static_cast<GradientType>(it->second * output_er);
    }
    gradients_touched_.push_back(idx);
  }
}

void NeuralNetConnection::FastUpdateWeightsMajor(float learning_rate) {
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
      vector<GradientType> &gt = gradients_[*it];
      vector<WeightType> &wt = weights_[*it];
      size_t d;
      for (d = 0; d + 7 < dim;) {
        WeightType &wt0 = wt[d];
        WeightType &wt1 = wt[d + 1];
        WeightType &wt2 = wt[d + 2];
        WeightType &wt3 = wt[d + 3];
        WeightType &wt4 = wt[d + 4];
        WeightType &wt5 = wt[d + 5];
        WeightType &wt6 = wt[d + 6];
        WeightType &wt7 = wt[d + 7];

        GradientType &gt0 = gt[d];
        GradientType &gt1 = gt[d + 1];
        GradientType &gt2 = gt[d + 2];
        GradientType &gt3 = gt[d + 3];
        GradientType &gt4 = gt[d + 4];
        GradientType &gt5 = gt[d + 5];
        GradientType &gt6 = gt[d + 6];
        GradientType &gt7 = gt[d + 7];

        const GradientType gt_l2_0 = gt0 - l2_regularization_param_ * wt0;
        const GradientType gt_l2_1 = gt1 - l2_regularization_param_ * wt1;
        const GradientType gt_l2_2 = gt2 - l2_regularization_param_ * wt2;
        const GradientType gt_l2_3 = gt3 - l2_regularization_param_ * wt3;
        const GradientType gt_l2_4 = gt4 - l2_regularization_param_ * wt4;
        const GradientType gt_l2_5 = gt5 - l2_regularization_param_ * wt5;
        const GradientType gt_l2_6 = gt6 - l2_regularization_param_ * wt6;
        const GradientType gt_l2_7 = gt7 - l2_regularization_param_ * wt7;

        wt0 += static_cast<WeightType>(learning_rate * gt_l2_0);
        wt1 += static_cast<WeightType>(learning_rate * gt_l2_1);
        wt2 += static_cast<WeightType>(learning_rate * gt_l2_2);
        wt3 += static_cast<WeightType>(learning_rate * gt_l2_3);
        wt4 += static_cast<WeightType>(learning_rate * gt_l2_4);
        wt5 += static_cast<WeightType>(learning_rate * gt_l2_5);
        wt6 += static_cast<WeightType>(learning_rate * gt_l2_6);
        wt7 += static_cast<WeightType>(learning_rate * gt_l2_7);

        // Clear gradients.
        gt0 = 0.0f;
        gt1 = 0.0f;
        gt2 = 0.0f;
        gt3 = 0.0f;
        gt4 = 0.0f;
        gt5 = 0.0f;
        gt6 = 0.0f;
        gt7 = 0.0f;

        d += 8;
      }
      for (; d < dim; d++) {
        WeightType &wt0 = wt[d];
        GradientType &gt0 = gt[d];
        const GradientType gt_l2_0 = gt0 - l2_regularization_param_ * wt0;
        wt0 += static_cast<WeightType>(learning_rate * gt_l2_0);
        // Clear gradients.
        gt0 = 0.0f;
      }

      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      }
    }
  } else {
    for (vector<size_t>::iterator it = gradients_touched_.begin();
         it != gradients_touched_.end(); ++it) {
      vector<GradientType> &gt = gradients_[*it];
      vector<GradientType> &var = sum_gradient_squares_[*it];
      vector<WeightType> &wt = weights_[*it];
      size_t d;
      for (d = 0; d + 7 < dim;) {
        WeightType &wt0 = wt[d];
        WeightType &wt1 = wt[d + 1];
        WeightType &wt2 = wt[d + 2];
        WeightType &wt3 = wt[d + 3];
        WeightType &wt4 = wt[d + 4];
        WeightType &wt5 = wt[d + 5];
        WeightType &wt6 = wt[d + 6];
        WeightType &wt7 = wt[d + 7];

        GradientType &gt0 = gt[d];
        GradientType &gt1 = gt[d + 1];
        GradientType &gt2 = gt[d + 2];
        GradientType &gt3 = gt[d + 3];
        GradientType &gt4 = gt[d + 4];
        GradientType &gt5 = gt[d + 5];
        GradientType &gt6 = gt[d + 6];
        GradientType &gt7 = gt[d + 7];

        const GradientType gt_l2_0 = gt0 - l2_regularization_param_ * wt0;
        const GradientType gt_l2_1 = gt1 - l2_regularization_param_ * wt1;
        const GradientType gt_l2_2 = gt2 - l2_regularization_param_ * wt2;
        const GradientType gt_l2_3 = gt3 - l2_regularization_param_ * wt3;
        const GradientType gt_l2_4 = gt4 - l2_regularization_param_ * wt4;
        const GradientType gt_l2_5 = gt5 - l2_regularization_param_ * wt5;
        const GradientType gt_l2_6 = gt6 - l2_regularization_param_ * wt6;
        const GradientType gt_l2_7 = gt7 - l2_regularization_param_ * wt7;

        GradientType &var0 = var[d];
        GradientType &var1 = var[d + 1];
        GradientType &var2 = var[d + 2];
        GradientType &var3 = var[d + 3];
        GradientType &var4 = var[d + 4];
        GradientType &var5 = var[d + 5];
        GradientType &var6 = var[d + 6];
        GradientType &var7 = var[d + 7];

        var0 += gt_l2_0 * gt_l2_0;
        var1 += gt_l2_1 * gt_l2_1;
        var2 += gt_l2_2 * gt_l2_2;
        var3 += gt_l2_3 * gt_l2_3;
        var4 += gt_l2_4 * gt_l2_4;
        var5 += gt_l2_5 * gt_l2_5;
        var6 += gt_l2_6 * gt_l2_6;
        var7 += gt_l2_7 * gt_l2_7;

        wt0 += static_cast<WeightType>(learning_rate * gt_l2_0 / sqrt(var0));
        wt1 += static_cast<WeightType>(learning_rate * gt_l2_1 / sqrt(var1));
        wt2 += static_cast<WeightType>(learning_rate * gt_l2_2 / sqrt(var2));
        wt3 += static_cast<WeightType>(learning_rate * gt_l2_3 / sqrt(var3));
        wt4 += static_cast<WeightType>(learning_rate * gt_l2_4 / sqrt(var4));
        wt5 += static_cast<WeightType>(learning_rate * gt_l2_5 / sqrt(var5));
        wt6 += static_cast<WeightType>(learning_rate * gt_l2_6 / sqrt(var6));
        wt7 += static_cast<WeightType>(learning_rate * gt_l2_7 / sqrt(var7));

        // Clear gradients.
        gt0 = 0.0f;
        gt1 = 0.0f;
        gt2 = 0.0f;
        gt3 = 0.0f;
        gt4 = 0.0f;
        gt5 = 0.0f;
        gt6 = 0.0f;
        gt7 = 0.0f;

        d += 8;
      }
      for (; d < dim; d++) {
        WeightType &wt0 = wt[d];
        GradientType &gt0 = gt[d];
        const GradientType gt_l2_0 = gt[d] - l2_regularization_param_ * wt0;
        GradientType &var0 = var[d];
        var0 += gt_l2_0 * gt_l2_0;
        wt0 += static_cast<WeightType>(learning_rate * gt_l2_0 / sqrt(var0));
        // Clear gradients.
        gt0 = 0.0f;
      }

      // Record lastupdate_.
      if (l2_regularization_param_ > 0) {
        lastupdate_[*it] = num_updates_;
      }
    }
  }

  // Clear gradients_touched_.
  gradients_touched_.clear();
}

void NeuralNetConnection::FastUpdateWeightsMinor() {
  assert(last_learning_rate_ > 0);

  if (l2_regularization_param_ > 0) {
    size_t dim1, dim2;
    if (storage_input_major_) {
      dim1 = ninput_;
      dim2 = noutput_;
    }
    else {
      dim1 = noutput_;
      dim2 = ninput_;
    }

    if (!adagrad_) {
      for (size_t d1 = 0; d1 < dim1; d1++) {
        int &lu = lastupdate_[d1];
        assert(lu <= num_updates_);
        vector<WeightType> &wt = weights_[d1];
        if (lu < num_updates_) {
          double scale = pow(1 - last_learning_rate_ * l2_regularization_param_, num_updates_ - lu);
          size_t d2;
          for (d2 = 0; d2 + 7 < dim2; ) {
            WeightType &wt0 = wt[d2];
            WeightType &wt1 = wt[d2 + 1];
            WeightType &wt2 = wt[d2 + 2];
            WeightType &wt3 = wt[d2 + 3];
            WeightType &wt4 = wt[d2 + 4];
            WeightType &wt5 = wt[d2 + 5];
            WeightType &wt6 = wt[d2 + 6];
            WeightType &wt7 = wt[d2 + 7];

            wt0 = static_cast<WeightType>(scale * wt0);
            wt1 = static_cast<WeightType>(scale * wt1);
            wt2 = static_cast<WeightType>(scale * wt2);
            wt3 = static_cast<WeightType>(scale * wt3);
            wt4 = static_cast<WeightType>(scale * wt4);
            wt5 = static_cast<WeightType>(scale * wt5);
            wt6 = static_cast<WeightType>(scale * wt6);
            wt7 = static_cast<WeightType>(scale * wt7);

            d2 += 8;
          }
          for (; d2 < dim2; d2++) {
            WeightType &wt0 = wt[d2];
            wt0 = static_cast<WeightType>(scale * wt0);
          }
        }
        lu = num_updates_;
      }
    } else {
      for (size_t d1 = 0; d1 < dim1; d1++) {
        int &lu = lastupdate_[d1];
        assert(lu <= num_updates_);
        vector<WeightType> &wt = weights_[d1];
        vector<GradientType> &var = sum_gradient_squares_[d1];
        for (int t = lu; t < num_updates_; t++) {
          size_t d2;
          for (d2 = 0; d2 + 7 < dim2; ) {
            WeightType &wt0 = wt[d2];
            WeightType &wt1 = wt[d2 + 1];
            WeightType &wt2 = wt[d2 + 2];
            WeightType &wt3 = wt[d2 + 3];
            WeightType &wt4 = wt[d2 + 4];
            WeightType &wt5 = wt[d2 + 5];
            WeightType &wt6 = wt[d2 + 6];
            WeightType &wt7 = wt[d2 + 7];

            const GradientType gt_l2_0 = -l2_regularization_param_ * wt0;
            const GradientType gt_l2_1 = -l2_regularization_param_ * wt1;
            const GradientType gt_l2_2 = -l2_regularization_param_ * wt2;
            const GradientType gt_l2_3 = -l2_regularization_param_ * wt3;
            const GradientType gt_l2_4 = -l2_regularization_param_ * wt4;
            const GradientType gt_l2_5 = -l2_regularization_param_ * wt5;
            const GradientType gt_l2_6 = -l2_regularization_param_ * wt6;
            const GradientType gt_l2_7 = -l2_regularization_param_ * wt7;

            GradientType &var0 = var[d2];
            GradientType &var1 = var[d2 + 1];
            GradientType &var2 = var[d2 + 2];
            GradientType &var3 = var[d2 + 3];
            GradientType &var4 = var[d2 + 4];
            GradientType &var5 = var[d2 + 5];
            GradientType &var6 = var[d2 + 6];
            GradientType &var7 = var[d2 + 7];

            var0 += gt_l2_0 * gt_l2_0;
            var1 += gt_l2_1 * gt_l2_1;
            var2 += gt_l2_2 * gt_l2_2;
            var3 += gt_l2_3 * gt_l2_3;
            var4 += gt_l2_4 * gt_l2_4;
            var5 += gt_l2_5 * gt_l2_5;
            var6 += gt_l2_6 * gt_l2_6;
            var7 += gt_l2_7 * gt_l2_7;

            wt0 += static_cast<WeightType>(last_learning_rate_ * gt_l2_0 / sqrt(var0));
            wt1 += static_cast<WeightType>(last_learning_rate_ * gt_l2_1 / sqrt(var1));
            wt2 += static_cast<WeightType>(last_learning_rate_ * gt_l2_2 / sqrt(var2));
            wt3 += static_cast<WeightType>(last_learning_rate_ * gt_l2_3 / sqrt(var3));
            wt4 += static_cast<WeightType>(last_learning_rate_ * gt_l2_4 / sqrt(var4));
            wt5 += static_cast<WeightType>(last_learning_rate_ * gt_l2_5 / sqrt(var5));
            wt6 += static_cast<WeightType>(last_learning_rate_ * gt_l2_6 / sqrt(var6));
            wt7 += static_cast<WeightType>(last_learning_rate_ * gt_l2_7 / sqrt(var7));

            d2 += 8;
          }
          for (; d2 < dim2; d2++) {
            WeightType &wt0 = wt[d2];
            const GradientType gt_l2_0 = -l2_regularization_param_ * wt0;
            GradientType &var0 = var[d2];
            var0 += gt_l2_0 * gt_l2_0;
            wt0 += static_cast<WeightType>(last_learning_rate_ * gt_l2_0 / sqrt(var0));
          }
        }
        lu = num_updates_;
      }
    }
  }
  // else, lastupdate_ is unused.

  last_learning_rate_ = -1;
}

void NeuralNetConnection::WriteConnectionToTxt(ostream &os) {
  os << "storage_input_major_: " << storage_input_major_ << endl;
  os << "ninput_: " << ninput_ << endl;
  os << "noutput_: " << noutput_ << endl;
  os << "l2_regularization_param_: " << l2_regularization_param_ << endl;
  size_t dim1, dim2;
  if (storage_input_major_) {
    dim1 = ninput_;
    dim2 = noutput_;
  } else {
    dim1 = noutput_;
    dim2 = ninput_;
  }
  for (size_t d1 = 0; d1 < dim1; d1++) {
    vector<WeightType> &wt = weights_[d1];
    for (size_t d2 = 0; d2 < dim2; d2++) {
      os << "weights_[" << d1 << "][" << d2 << "]: " << wt[d2] << endl;
    }
  }
}

void NeuralNetConnection::AllocateConnection() {
  size_t dim1, dim2;
  if (storage_input_major_) {
    dim1 = ninput_;
    dim2 = noutput_;
  } else {
    dim1 = noutput_;
    dim2 = ninput_;
  }

  weights_.resize(dim1);
  for (size_t d1 = 0; d1 < dim1; d1++) {
    weights_[d1].resize(dim2);
    fill(weights_[d1].begin(), weights_[d1].end(), 0.0f);
  }

  gradients_.resize(dim1);
  for (size_t d1 = 0; d1 < dim1; d1++) {
    gradients_[d1].resize(dim2);
    fill(gradients_[d1].begin(), gradients_[d1].end(), 0.0f);
  }

  sum_gradient_squares_.resize(dim1);
  for (size_t d1 = 0; d1 < dim1; d1++) {
    sum_gradient_squares_[d1].resize(dim2);
    // Note: Empirically, it is good to initialize it as 1.0f. If something
    // weired happens, come here and check the value.
    // The idea is to make sure the approximated Hessian is invertible (check
    // original paper).
    fill(sum_gradient_squares_[d1].begin(), sum_gradient_squares_[d1].end(), 1.0f);
  }

  lastupdate_.resize(dim1);
  fill(lastupdate_.begin(), lastupdate_.end(), 0);

  gradients_touched_.clear();

  num_updates_ = 0;
  last_learning_rate_ = -1;
}

void NeuralNetConnection::FastUpdateWeightsMinor(size_t idx) {
  assert(last_learning_rate_ > 0);
  assert(lastupdate_[idx] < num_updates_);
  assert(l2_regularization_param_ > 0);

  vector<WeightType> &wt = weights_[idx];
  size_t dim;
  if (storage_input_major_) {
    dim = noutput_;
  } else {
    dim = ninput_;
  }
  assert(idx < dim);

  if (!adagrad_) {
    double scale = pow(1 - last_learning_rate_ * l2_regularization_param_, num_updates_ - lastupdate_[idx]);

    size_t d;
    for (d = 0; d + 7 < dim; ) {
      WeightType &wt0 = wt[d];
      WeightType &wt1 = wt[d + 1];
      WeightType &wt2 = wt[d + 2];
      WeightType &wt3 = wt[d + 3];
      WeightType &wt4 = wt[d + 4];
      WeightType &wt5 = wt[d + 5];
      WeightType &wt6 = wt[d + 6];
      WeightType &wt7 = wt[d + 7];

      wt0 = static_cast<WeightType>(scale * wt0);
      wt1 = static_cast<WeightType>(scale * wt1);
      wt2 = static_cast<WeightType>(scale * wt2);
      wt3 = static_cast<WeightType>(scale * wt3);
      wt4 = static_cast<WeightType>(scale * wt4);
      wt5 = static_cast<WeightType>(scale * wt5);
      wt6 = static_cast<WeightType>(scale * wt6);
      wt7 = static_cast<WeightType>(scale * wt7);

      d += 8;
    }
    for (; d < dim; d++) {
      WeightType &wt0 = wt[d];
      wt0 = static_cast<WeightType>(scale * wt0);
    } 
  } else {
    vector<GradientType> &var = sum_gradient_squares_[idx];
    // Note: should exchange loop and check if wt == 0? Usually num_updates_ is
    // much larger (== num_tokens_).
    for (int t = lastupdate_[idx]; t < num_updates_; t++) {
      size_t d;
      for (d = 0; d + 7 < dim;) {
        WeightType &wt0 = wt[d];
        WeightType &wt1 = wt[d + 1];
        WeightType &wt2 = wt[d + 2];
        WeightType &wt3 = wt[d + 3];
        WeightType &wt4 = wt[d + 4];
        WeightType &wt5 = wt[d + 5];
        WeightType &wt6 = wt[d + 6];
        WeightType &wt7 = wt[d + 7];

        const GradientType gt_l2_0 = -l2_regularization_param_ * wt0;
        const GradientType gt_l2_1 = -l2_regularization_param_ * wt1;
        const GradientType gt_l2_2 = -l2_regularization_param_ * wt2;
        const GradientType gt_l2_3 = -l2_regularization_param_ * wt3;
        const GradientType gt_l2_4 = -l2_regularization_param_ * wt4;
        const GradientType gt_l2_5 = -l2_regularization_param_ * wt5;
        const GradientType gt_l2_6 = -l2_regularization_param_ * wt6;
        const GradientType gt_l2_7 = -l2_regularization_param_ * wt7;

        GradientType &var0 = var[d];
        GradientType &var1 = var[d + 1];
        GradientType &var2 = var[d + 2];
        GradientType &var3 = var[d + 3];
        GradientType &var4 = var[d + 4];
        GradientType &var5 = var[d + 5];
        GradientType &var6 = var[d + 6];
        GradientType &var7 = var[d + 7];

        var0 += gt_l2_0 * gt_l2_0;
        var1 += gt_l2_1 * gt_l2_1;
        var2 += gt_l2_2 * gt_l2_2;
        var3 += gt_l2_3 * gt_l2_3;
        var4 += gt_l2_4 * gt_l2_4;
        var5 += gt_l2_5 * gt_l2_5;
        var6 += gt_l2_6 * gt_l2_6;
        var7 += gt_l2_7 * gt_l2_7;

        wt0 += static_cast<WeightType>(last_learning_rate_ * gt_l2_0 / sqrt(var0));
        wt1 += static_cast<WeightType>(last_learning_rate_ * gt_l2_1 / sqrt(var1));
        wt2 += static_cast<WeightType>(last_learning_rate_ * gt_l2_2 / sqrt(var2));
        wt3 += static_cast<WeightType>(last_learning_rate_ * gt_l2_3 / sqrt(var3));
        wt4 += static_cast<WeightType>(last_learning_rate_ * gt_l2_4 / sqrt(var4));
        wt5 += static_cast<WeightType>(last_learning_rate_ * gt_l2_5 / sqrt(var5));
        wt6 += static_cast<WeightType>(last_learning_rate_ * gt_l2_6 / sqrt(var6));
        wt7 += static_cast<WeightType>(last_learning_rate_ * gt_l2_7 / sqrt(var7));

        d += 8;
      }
      for (; d < dim; d++) {
        WeightType &wt0 = wt[d];
        const GradientType gt_l2_0 = -l2_regularization_param_ * wt0;
        GradientType &var0 = var[d];
        var0 += gt_l2_0 * gt_l2_0;
        wt0 += static_cast<WeightType>(last_learning_rate_ * gt_l2_0 / sqrt(var0));
      }
    }
  }
  lastupdate_[idx] = num_updates_;
}

} // namespace neuralnet
