#ifndef NEURALNET_FULL_CIRCULAR_BUFFER_H_
#define NEURALNET_FULL_CIRCULAR_BUFFER_H_

#include <cstdio>
#include <iostream>
#include <vector>

namespace neuralnet {

// A simple full circular buffer.
// Alpha version.
// Very limited function support for neural network back propagation through
// time algorithm.
template <class T>
class FullCircularBuffer {
 public:
  FullCircularBuffer() {
    capacity_ = 0;
    offset_ = 0;
  }

  ~FullCircularBuffer() {
  }

  // Sets the capacity.
  // Automatically calls AllocateBuffer().
  void set_capacity(std::size_t c) {
    capacity_ = c;
    AllocateBuffer();
  }

  // Sets the offset.
  void set_offset(std::size_t offset) {
    if (offset >= capacity_) {
      std::cerr << "index out of circular buffer capacity!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    offset_ = offset;
  }

  // Rotates the buffer.
  // The first + offset_ becomes the new offset (the new top of the buffer)
  void rotate(std::size_t first) {
    if (first >= capacity_) {
      std::cerr << "index out of circular buffer capacity!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (first + offset_ >= capacity_) {
      offset_ = first + offset_ - capacity_;
    } else {
      offset_ = first + offset_;
    }
  }

  T &operator[] (std::size_t idx) {
    if (idx >= capacity_) {
      std::cerr << "index out of circular buffer capacity!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (idx + offset_ >= capacity_) {
      return circular_buffer_[idx + offset_ - capacity_];
    } else {
      return circular_buffer_[idx + offset_];
    }
  }

  const T &operator[] (std::size_t idx) const {
    if (idx >= capacity_) {
      std::cerr << "index out of circular buffer capacity!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (idx + offset_ >= capacity_) {
      return circular_buffer_[idx + offset_ - capacity_];
    } else {
      return circular_buffer_[idx + offset_];
    }
  }

 private:
  // Allocates the buffer.
  void AllocateBuffer() {
    if (capacity_ == 0) {
      std::cerr << "capacity_ == 0!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    circular_buffer_.resize(capacity_);
    offset_ = 0;
  }

  std::vector<T> circular_buffer_;
  std::size_t offset_;
  std::size_t capacity_;
};

} // namespace neuralnet

#endif
