#ifndef NEURALNET_NEURALNET_NUMERIC_H_
#define NEURALNET_NEURALNET_NUMERIC_H_

#include <cmath>
#include <algorithm>

namespace neuralnet {

// Mikolov's fastexp implementation
static union {
  double d;
  struct {
    int j, i;
  } n;
} d2i;

template<class T>
inline double stable_fast_exp(T y) {
  // for numerical stability
  if (y > 50) y = 50;
  if (y < -50) y = -50;

  d2i.n.i = static_cast<int>(1048576 / 0.69314718055994530942 * y + (1072693248-60801));
  return d2i.d;
}

// target = log(x)
// inc = log(y)
// exp(target) = (x + y)/x if x > y
// exp(target) = (x + y)/y if x <=y
// i.e., 1 < exp(target) <= 2
// This is used for numerical stability consideration.
// If inc is too big, directly compute exp can be disaster.
inline void logadd(double &target, double inc) {
  if (target < inc) std::swap(target, inc);
  target += std::log(1 + std::exp(inc - target));
}

const unsigned int PRIMES[] = {108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
	407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
	782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int PRIMES_SIZE = sizeof(PRIMES)/sizeof(PRIMES[0]);
const unsigned int HASH_OFFSET = PRIMES[0] * PRIMES[1];

inline void hash_update0(std::size_t &hashval, const std::size_t nv) {
  hashval = hashval * PRIMES[nv % PRIMES_SIZE] + nv;
}

inline void hash_update1(std::size_t &hashval, const std::size_t a, const std::size_t b, const std::size_t nv) {
  hashval += PRIMES[(a*PRIMES[b] + b) % PRIMES_SIZE] * (unsigned long long)(nv + 1);
}

} // namespace neuralnet

#endif
