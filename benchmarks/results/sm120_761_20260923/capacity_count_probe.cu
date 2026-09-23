#include <cuco/static_multiset.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>

template <class Set>
double count_us(Set& set, thrust::device_vector<int> const& keys, std::size_t expected)
{
  auto start = std::chrono::steady_clock::now();
  auto count = set.count(keys.begin(), keys.end());
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  if (count != expected) {
    std::cerr << "count mismatch: " << count << " != " << expected << '\n';
    std::exit(1);
  }
  return std::chrono::duration<double, std::micro>(end - start).count();
}

int main()
{
  constexpr std::size_t n = 1 << 20;
  thrust::device_vector<int> hits(n);
  thrust::device_vector<int> misses(n);
  thrust::sequence(hits.begin(), hits.end());
  thrust::sequence(misses.begin(), misses.end(), static_cast<int>(n));

  cuco::static_multiset<int> low{2 * n, cuco::empty_key{-1}};
  cuco::static_multiset<int> high{4 * n, cuco::empty_key{-1}};
  low.insert(hits.begin(), hits.end());
  high.insert(hits.begin(), hits.end());
  std::cout << "capacity_low," << low.capacity() << "\ncapacity_high," << high.capacity()
            << "\nprobe,pair,low_us,high_us\n";

  for (int probe = 0; probe < 2; ++probe) {
    auto const& keys = probe == 0 ? hits : misses;
    std::size_t expected = probe == 0 ? n : 0;
    for (int warmup = 0; warmup < 4; ++warmup) {
      count_us(low, keys, expected);
      count_us(high, keys, expected);
    }
    for (int pair = 0; pair < 24; ++pair) {
      double low_us;
      double high_us;
      if (pair % 2 == 0) {
        low_us = count_us(low, keys, expected);
        high_us = count_us(high, keys, expected);
      } else {
        high_us = count_us(high, keys, expected);
        low_us = count_us(low, keys, expected);
      }
      std::cout << (probe == 0 ? "hit" : "miss") << ',' << pair << ',' << low_us << ','
                << high_us << '\n';
    }
  }
}
