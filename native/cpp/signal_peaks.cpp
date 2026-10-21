// Peak finding: scipy.signal._peak_finding_utils (_local_maxima_1d, _select_by_peak_distance, _peak_prominences,
// _peak_widths), ported from scipy 1.16 (BSD-3-Clause, Copyright (c) 2003-2024 SciPy Developers).
// Difference: _select_by_peak_distance orders equal-height peaks with a stable sort (numpy's default argsort is not
// stable, so scipy's choice between exactly equal heights is unspecified).
#include "signal_impl.hpp"
#include "host.hpp"
#include <algorithm>
#include <numeric>

namespace lmpy {
namespace sig {

void local_maxima_1d(const V& x, std::vector<long>& mid, std::vector<long>& left, std::vector<long>& right) {
  mid.clear(); left.clear(); right.clear();
  long n = long(x.size()), i = 1, i_max = n - 1;
  while (i < i_max) {
    if (x[size_t(i - 1)] < x[size_t(i)]) {
      long ahead = i + 1;
      while (ahead < i_max && x[size_t(ahead)] == x[size_t(i)]) ++ahead;
      if (x[size_t(ahead)] < x[size_t(i)]) {
        left.push_back(i);
        right.push_back(ahead - 1);
        mid.push_back((i + ahead - 1) / 2);
        i = ahead;
      }
    }
    ++i;
  }
}

std::vector<char> select_by_peak_distance(const std::vector<long>& peaks, const V& priority, double distance) {
  long size = long(peaks.size());
  long dist = long(std::ceil(distance));
  std::vector<char> keep(size_t(size), 1);
  std::vector<size_t> order(static_cast<size_t>(size));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return priority[a] < priority[b]; });
  for (long i = size - 1; i >= 0; --i) {
    long j = long(order[size_t(i)]);
    if (!keep[size_t(j)]) continue;
    long k = j - 1;
    while (0 <= k && peaks[size_t(j)] - peaks[size_t(k)] < dist) { keep[size_t(k)] = 0; --k; }
    k = j + 1;
    while (k < size && peaks[size_t(k)] - peaks[size_t(j)] < dist) { keep[size_t(k)] = 0; ++k; }
  }
  return keep;
}

void peak_prominences(const V& x, const std::vector<long>& peaks, long wlen, V& prom, std::vector<long>& lb, std::vector<long>& rb) {
  size_t np = peaks.size();
  prom.assign(np, 0); lb.assign(np, 0); rb.assign(np, 0);
  for (size_t p = 0; p < np; ++p) {
    long peak = peaks[p], i_min = 0, i_max = long(x.size()) - 1;
    if (!(i_min <= peak && peak <= i_max)) fail("peak_prominences: peak " + std::to_string(peak) + " is not a valid index for x");
    if (2 <= wlen) {
      i_min = std::max(peak - wlen / 2, i_min);
      i_max = std::min(peak + wlen / 2, i_max);
    }
    long i = lb[p] = peak;
    double left_min = x[size_t(peak)];
    while (i_min <= i && x[size_t(i)] <= x[size_t(peak)]) {
      if (x[size_t(i)] < left_min) { left_min = x[size_t(i)]; lb[p] = i; }
      --i;
    }
    i = rb[p] = peak;
    double right_min = x[size_t(peak)];
    while (i <= i_max && x[size_t(i)] <= x[size_t(peak)]) {
      if (x[size_t(i)] < right_min) { right_min = x[size_t(i)]; rb[p] = i; }
      ++i;
    }
    prom[p] = x[size_t(peak)] - std::max(left_min, right_min);
  }
}

void peak_widths(const V& x, const std::vector<long>& peaks, double rel_height, const V& prom, const std::vector<long>& lb,
                 const std::vector<long>& rb, V& widths, V& heights, V& lips, V& rips) {
  if (rel_height < 0) fail("peak_widths: rel_height must be greater or equal to 0");
  size_t np = peaks.size();
  widths.assign(np, 0); heights.assign(np, 0); lips.assign(np, 0); rips.assign(np, 0);
  for (size_t p = 0; p < np; ++p) {
    long i_min = lb[p], i_max = rb[p], peak = peaks[p];
    if (!(0 <= i_min && i_min <= peak && peak <= i_max && i_max < long(x.size())))
      fail("peak_widths: prominence data is invalid for peak " + std::to_string(peak));
    double height = heights[p] = x[size_t(peak)] - prom[p] * rel_height;
    long i = peak;
    while (i_min < i && height < x[size_t(i)]) --i;
    double left_ip = double(i);
    if (x[size_t(i)] < height) left_ip += (height - x[size_t(i)]) / (x[size_t(i + 1)] - x[size_t(i)]);
    i = peak;
    while (i < i_max && height < x[size_t(i)]) ++i;
    double right_ip = double(i);
    if (x[size_t(i)] < height) right_ip -= (height - x[size_t(i)]) / (x[size_t(i - 1)] - x[size_t(i)]);
    widths[p] = right_ip - left_ip;
    lips[p] = left_ip;
    rips[p] = right_ip;
  }
}

}  // namespace sig
}  // namespace lmpy
