// One-dimensional adaptive quadrature behind `quad`: QUADPACK's DQAGSE (finite interval, 21-point Gauss-Kronrod with
// Wynn's epsilon extrapolation), DQAGIE (infinite range mapped to (0, 1], 15-point rule) and DQAGPE (user break points),
// with DQK21, DQK15I, DQELG and DQPSRT, re-implemented from the published algorithms (R. Piessens, E. de Doncker-Kapenga,
// C. Uberhuber, D. Kahaner, "QUADPACK: A Subroutine Package for Automatic Integration", Springer 1983; QUADPACK is in the
// public domain), as wrapped by scipy.integrate.quad. Gauss-Kronrod nodes and weights come from Boost.Math (BSL-1.0).
#include "optimize_impl.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <algorithm>
#include <cmath>

namespace lmnum::opt {
namespace {

const double epmach = EPS;
const double uflow = std::numeric_limits<double>::min();
const double oflow = std::numeric_limits<double>::max();
using F1 = std::function<double(double)>;

// 21-point Gauss-Kronrod on [a, b] (DQK21)
void qk21(const F1& f, double a, double b, double& result, double& abserr, double& resabs, double& resasc) {
  using GK = boost::math::quadrature::gauss_kronrod<double, 21>;
  using G = boost::math::quadrature::gauss<double, 10>;
  const auto& xk = GK::abscissa();  // 0, then increasing; odd entries are the Gauss nodes
  const auto& wk = GK::weights();
  const auto& wg = G::weights();    // Gauss weights for abscissa 0.1488.., 0.4333.., ...
  double centr = 0.5 * (a + b), hlgth = 0.5 * (b - a), dhlgth = std::fabs(hlgth);
  double fc = f(centr);
  double resg = 0, resk = wk[0] * fc;
  resabs = std::fabs(resk);
  double fv1[11], fv2[11];
  for (int j = 1; j <= 10; ++j) {
    double absc = hlgth * xk[size_t(j)];
    double f1 = f(centr - absc), f2 = f(centr + absc);
    fv1[j] = f1; fv2[j] = f2;
    double fsum = f1 + f2;
    if (j % 2 == 1) resg += wg[size_t(j / 2)] * fsum;
    resk += wk[size_t(j)] * fsum;
    resabs += wk[size_t(j)] * (std::fabs(f1) + std::fabs(f2));
  }
  double reskh = resk * 0.5;
  resasc = wk[0] * std::fabs(fc - reskh);
  for (int j = 1; j <= 10; ++j) resasc += wk[size_t(j)] * (std::fabs(fv1[j] - reskh) + std::fabs(fv2[j] - reskh));
  result = resk * hlgth;
  resabs *= dhlgth;
  resasc *= dhlgth;
  abserr = std::fabs((resk - resg) * hlgth);
  if (resasc != 0 && abserr != 0) abserr = resasc * std::min(1.0, std::pow(200 * abserr / resasc, 1.5));
  if (resabs > uflow / (50 * epmach)) abserr = std::max((epmach * 50) * resabs, abserr);
}

// 15-point transformed rule for (bound, +-inf) mapped to (0, 1] (DQK15I)
void qk15i(const F1& f, double boun, int inf, double a, double b, double& result, double& abserr, double& resabs, double& resasc) {
  using GK = boost::math::quadrature::gauss_kronrod<double, 15>;
  using G = boost::math::quadrature::gauss<double, 7>;
  const auto& xk = GK::abscissa();  // 0, 0.2077.., 0.4058.., ... ; odd entries are 7-point Gauss nodes (with 0)
  const auto& wk = GK::weights();
  const auto& wg = G::weights();    // wg[0] centre, wg[1..3] for 0.4058.., 0.7415.., 0.9491..
  double dinf = std::min(1, inf);
  double centr = 0.5 * (a + b), hlgth = 0.5 * (b - a);
  auto g = [&](double t) {
    double x = boun + dinf * (1 - t) / t;
    double v = f(x);
    if (inf == 2) v += f(-x);
    return (v / t) / t;
  };
  double fc = g(centr);
  double resg = wg[0] * fc, resk = wk[0] * fc;
  resabs = std::fabs(resk);
  double fv1[8], fv2[8];
  for (int j = 1; j <= 7; ++j) {
    double absc = hlgth * xk[size_t(j)];
    double f1 = g(centr - absc), f2 = g(centr + absc);
    fv1[j] = f1; fv2[j] = f2;
    double fsum = f1 + f2;
    if (j % 2 == 0) resg += wg[size_t(j / 2)] * fsum;
    resk += wk[size_t(j)] * fsum;
    resabs += wk[size_t(j)] * (std::fabs(f1) + std::fabs(f2));
  }
  double reskh = resk * 0.5;
  resasc = wk[0] * std::fabs(fc - reskh);
  for (int j = 1; j <= 7; ++j) resasc += wk[size_t(j)] * (std::fabs(fv1[j] - reskh) + std::fabs(fv2[j] - reskh));
  result = resk * hlgth;
  resasc *= hlgth;
  resabs *= hlgth;
  abserr = std::fabs((resk - resg) * hlgth);
  if (resasc != 0 && abserr != 0) abserr = resasc * std::min(1.0, std::pow(200 * abserr / resasc, 1.5));
  if (resabs > uflow / (50 * epmach)) abserr = std::max((epmach * 50) * resabs, abserr);
}

// Wynn's epsilon algorithm (DQELG). Arrays are 1-based to follow the published algorithm.
void qelg(int& n, double* epstab, double& result, double& abserr, double* res3la, int& nres) {
  ++nres;
  abserr = oflow;
  result = epstab[n];
  if (n < 3) { abserr = std::max(abserr, 5 * epmach * std::fabs(result)); return; }
  const int limexp = 50;
  epstab[n + 2] = epstab[n];
  int newelm = (n - 1) / 2;
  epstab[n] = oflow;
  int num = n, k1 = n;
  bool converged = false;
  for (int i = 1; i <= newelm; ++i) {
    int k2 = k1 - 1, k3 = k1 - 2;
    double res = epstab[k1 + 2], e0 = epstab[k3], e1 = epstab[k2], e2 = res;
    double e1abs = std::fabs(e1), delta2 = e2 - e1, err2 = std::fabs(delta2);
    double tol2 = std::max(std::fabs(e2), e1abs) * epmach;
    double delta3 = e1 - e0, err3 = std::fabs(delta3), tol3 = std::max(e1abs, std::fabs(e0)) * epmach;
    if (err2 <= tol2 && err3 <= tol3) {
      result = res;
      abserr = err2 + err3;
      converged = true;
      break;
    }
    double e3 = epstab[k1];
    epstab[k1] = e1;
    double delta1 = e1 - e3, err1 = std::fabs(delta1), tol1 = std::max(e1abs, std::fabs(e3)) * epmach;
    if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) { n = i + i - 1; break; }
    double ss = 1 / delta1 + 1 / delta2 - 1 / delta3;
    double epsinf = std::fabs(ss * e1);
    if (!(epsinf > 1e-4)) { n = i + i - 1; break; }
    res = e1 + 1 / ss;
    epstab[k1] = res;
    k1 -= 2;
    double error = err2 + std::fabs(res - e2) + err3;
    if (error > abserr) continue;
    abserr = error;
    result = res;
  }
  if (!converged) {
    if (n == limexp) n = 2 * (limexp / 2) - 1;
    int ib = (num % 2 == 0) ? 2 : 1;
    int ie = newelm + 1;
    for (int i = 1; i <= ie; ++i) {
      int ib2 = ib + 2;
      epstab[ib] = epstab[ib2];
      ib = ib2;
    }
    if (num != n) {
      int indx = num - n + 1;
      for (int i = 1; i <= n; ++i) { epstab[i] = epstab[indx]; ++indx; }
    }
    if (nres < 4) {
      res3la[nres] = result;
      abserr = oflow;
    } else {
      abserr = std::fabs(result - res3la[3]) + std::fabs(result - res3la[2]) + std::fabs(result - res3la[1]);
      res3la[1] = res3la[2];
      res3la[2] = res3la[3];
      res3la[3] = result;
    }
  }
  abserr = std::max(abserr, 5 * epmach * std::fabs(result));
}

// DQPSRT: maintain the descending ordering of error estimates (1-based arrays)
void qpsrt(int limit, int last, int& maxerr, double& ermax, const double* elist, int* iord, int& nrmax) {
  if (last <= 2) {
    iord[1] = 1;
    iord[2] = 2;
  } else {
    double errmax = elist[maxerr];
    if (nrmax != 1) {
      int ido = nrmax - 1;
      for (int i = 1; i <= ido; ++i) {
        int isucc = iord[nrmax - 1];
        if (errmax <= elist[isucc]) break;
        iord[nrmax] = isucc;
        --nrmax;
      }
    }
    int jupbn = last;
    if (last > (limit / 2 + 2)) jupbn = limit + 3 - last;
    double errmin = elist[last];
    int jbnd = jupbn - 1, ibeg = nrmax + 1;
    bool inserted = false;
    int i = ibeg;
    for (; i <= jbnd; ++i) {
      int isucc = iord[i];
      if (errmax >= elist[isucc]) { inserted = true; break; }
      iord[i - 1] = isucc;
    }
    if (!inserted) {
      iord[jbnd] = maxerr;
      iord[jupbn] = last;
    } else {
      iord[i - 1] = maxerr;
      int k = jbnd;
      bool placed = false;
      for (int j = i; j <= jbnd; ++j) {
        int isucc = iord[k];
        if (errmin < elist[isucc]) { iord[k + 1] = last; placed = true; break; }
        iord[k + 1] = isucc;
        --k;
      }
      if (!placed) iord[i] = last;
    }
  }
  maxerr = iord[nrmax];
  ermax = elist[maxerr];
}

// Shared body of DQAGSE / DQAGIE: rule(a, b, result, abserr, resabs, resasc)
template <class Rule>
QuadResult qags_core(Rule rule, double a, double b, double epsabs, double epsrel, int limit, int evals_per_rule, int neval_mult) {
  QuadResult R;
  std::vector<double> alist(size_t(limit) + 2), blist(size_t(limit) + 2), rlist(size_t(limit) + 2), elist(size_t(limit) + 2);
  std::vector<int> iord(size_t(limit) + 2);
  double rlist2[55] = {}, res3la[4] = {};
  int ier = 0, last = 0;
  double result = 0, abserr = 0;
  alist[1] = a; blist[1] = b; rlist[1] = 0; elist[1] = 0;
  if (epsabs <= 0 && epsrel < std::max(50 * epmach, 0.5e-28)) { R.ier = 6; return R; }
  int ierro = 0;
  double defabs, resabs;
  rule(a, b, result, abserr, defabs, resabs);
  double dres = std::fabs(result), errbnd = std::max(epsabs, epsrel * dres);
  last = 1;
  rlist[1] = result; elist[1] = abserr; iord[1] = 1;
  if (abserr <= 100 * epmach * defabs && abserr > errbnd) ier = 2;
  if (limit == 1) ier = 1;
  auto finish = [&](int lastv) {
    R.value = result; R.abserr = abserr; R.ier = ier; R.last = lastv;
    R.neval = int64_t(evals_per_rule) * (2 * lastv - 1) * neval_mult;
    return R;
  };
  if (ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0) return finish(last);
  rlist2[1] = result;
  double errmax = abserr, area = result, errsum = abserr, small = 0, erlarg = 0, ertest = 0, correc = 0;
  int maxerr = 1;
  abserr = oflow;
  int nrmax = 1, nres = 0, numrl2 = 2, ktmin = 0;
  bool extrap = false, noext = false;
  int iroff1 = 0, iroff2 = 0, iroff3 = 0;
  int ksgn = -1;
  if (dres >= (1 - 50 * epmach) * defabs) ksgn = 1;
  enum { NONE, SUM_RESULT } tail = NONE;
  bool goto100 = false;
  for (last = 2; last <= limit; ++last) {
    double a1 = alist[size_t(maxerr)], b1 = 0.5 * (alist[size_t(maxerr)] + blist[size_t(maxerr)]), a2 = b1, b2 = blist[size_t(maxerr)];
    double erlast = errmax, area1, error1, defab1, area2, error2, defab2, ra;
    rule(a1, b1, area1, error1, ra, defab1);
    rule(a2, b2, area2, error2, ra, defab2);
    double area12 = area1 + area2, erro12 = error1 + error2;
    errsum = errsum + erro12 - errmax;
    area = area + area12 - rlist[size_t(maxerr)];
    if (!(defab1 == error1 || defab2 == error2)) {
      if (!(std::fabs(rlist[size_t(maxerr)] - area12) > 1e-5 * std::fabs(area12) || erro12 < 0.99 * errmax)) {
        if (extrap) ++iroff2;
        else ++iroff1;
      }
      if (last > 10 && erro12 > errmax) ++iroff3;
    }
    rlist[size_t(maxerr)] = area1;
    rlist[size_t(last)] = area2;
    errbnd = std::max(epsabs, epsrel * std::fabs(area));
    if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
    if (iroff2 >= 5) ierro = 3;
    if (last == limit) ier = 1;
    if (std::max(std::fabs(a1), std::fabs(b2)) <= (1 + 100 * epmach) * (std::fabs(a2) + 1000 * uflow)) ier = 4;
    if (error2 > error1) {
      alist[size_t(maxerr)] = a2; alist[size_t(last)] = a1; blist[size_t(last)] = b1;
      rlist[size_t(maxerr)] = area2; rlist[size_t(last)] = area1;
      elist[size_t(maxerr)] = error2; elist[size_t(last)] = error1;
    } else {
      alist[size_t(last)] = a2; blist[size_t(maxerr)] = b1; blist[size_t(last)] = b2;
      elist[size_t(maxerr)] = error1; elist[size_t(last)] = error2;
    }
    qpsrt(limit, last, maxerr, errmax, elist.data(), iord.data(), nrmax);
    if (errsum <= errbnd) { tail = SUM_RESULT; break; }
    if (ier != 0) { goto100 = true; break; }
    if (last == 2) {
      small = std::fabs(b - a) * 0.375;
      erlarg = errsum;
      ertest = errbnd;
      rlist2[2] = area;
      continue;
    }
    if (noext) continue;
    erlarg -= erlast;
    if (std::fabs(b1 - a1) > small) erlarg += erro12;
    if (!extrap) {
      if (std::fabs(blist[size_t(maxerr)] - alist[size_t(maxerr)]) > small) continue;
      extrap = true;
      nrmax = 2;
    }
    bool do_extrap = true;
    if (!(ierro == 3 || erlarg <= ertest)) {
      int id = nrmax, jupbnd = last;
      if (last > (2 + limit / 2)) jupbnd = limit + 3 - last;
      bool jumped = false;
      for (int k = id; k <= jupbnd; ++k) {
        maxerr = iord[size_t(nrmax)];
        errmax = elist[size_t(maxerr)];
        if (std::fabs(blist[size_t(maxerr)] - alist[size_t(maxerr)]) > small) { jumped = true; break; }
        ++nrmax;
      }
      if (jumped) do_extrap = false;
    }
    if (!do_extrap) continue;
    ++numrl2;
    rlist2[numrl2] = area;
    double reseps, abseps;
    qelg(numrl2, rlist2, reseps, abseps, res3la, nres);
    ++ktmin;
    if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
    if (abseps < abserr) {
      ktmin = 0;
      abserr = abseps;
      result = reseps;
      correc = erlarg;
      ertest = std::max(epsabs, epsrel * std::fabs(reseps));
      if (abserr <= ertest) { goto100 = true; break; }
    }
    if (numrl2 == 1) noext = true;
    if (ier == 5) { goto100 = true; break; }
    maxerr = iord[1];
    errmax = elist[size_t(maxerr)];
    nrmax = 1;
    extrap = false;
    small *= 0.5;
    erlarg = errsum;
  }
  if (last > limit) last = limit;
  (void)goto100;
  // label 100: final result
  if (tail != SUM_RESULT) {
    bool sum_it = false, divergence_test = false;
    if (abserr == oflow) sum_it = true;
    else if (ier + ierro == 0) divergence_test = true;
    else {
      if (ierro == 3) abserr += correc;
      if (ier == 0) ier = 3;
      if (result != 0 && area != 0) {
        if (abserr / std::fabs(result) > errsum / std::fabs(area)) sum_it = true;
        else divergence_test = true;
      } else if (abserr > errsum) sum_it = true;
      else if (area == 0) { /* goto 130 */ }
      else divergence_test = true;
    }
    if (divergence_test) {
      if (!(ksgn == -1 && std::max(std::fabs(result), std::fabs(area)) <= defabs * 0.01))
        if (0.01 > (result / area) || (result / area) > 100 || errsum > std::fabs(area)) ier = 6;
    }
    if (sum_it) tail = SUM_RESULT;
  }
  if (tail == SUM_RESULT) {
    result = 0;
    for (int k = 1; k <= last; ++k) result += rlist[size_t(k)];
    abserr = errsum;
  }
  if (ier > 2) --ier;
  return finish(last);
}

QuadResult qagp(const F1& f, double a, double b, std::vector<double> points, double epsabs, double epsrel, int limit) {
  QuadResult R;
  int npts = int(points.size()), npts2 = npts + 2;
  if (limit <= npts || (epsabs <= 0 && epsrel < std::max(50 * epmach, 0.5e-28))) { R.ier = 6; return R; }
  double sign = a > b ? -1 : 1;
  std::vector<double> pts(size_t(npts2) + 1);
  pts[1] = std::min(a, b);
  for (int i = 1; i <= npts; ++i) pts[size_t(i + 1)] = points[size_t(i - 1)];
  pts[size_t(npts + 2)] = std::max(a, b);
  int nint = npts + 1;
  std::sort(pts.begin() + 1, pts.end());
  std::vector<double> alist(size_t(limit) + 2), blist(size_t(limit) + 2), rlist(size_t(limit) + 2), elist(size_t(limit) + 2);
  std::vector<int> iord(size_t(limit) + 2), level(size_t(limit) + 2), ndin(size_t(npts2) + 2);
  double rlist2[55] = {}, res3la[4] = {};
  double result = 0, abserr = 0, resabs = 0, a1 = pts[1];
  int ier = 0;
  for (int i = 1; i <= nint; ++i) {
    double b1 = pts[size_t(i + 1)], area1, error1, defabs, resa;
    qk21(f, a1, b1, area1, error1, defabs, resa);
    abserr += error1;
    result += area1;
    ndin[size_t(i)] = 0;
    if (error1 == resa && error1 != 0) ndin[size_t(i)] = 1;
    resabs += defabs;
    level[size_t(i)] = 0;
    elist[size_t(i)] = error1; alist[size_t(i)] = a1; blist[size_t(i)] = b1; rlist[size_t(i)] = area1;
    iord[size_t(i)] = i;
    a1 = b1;
  }
  double errsum = 0;
  for (int i = 1; i <= nint; ++i) {
    if (ndin[size_t(i)] == 1) elist[size_t(i)] = abserr;
    errsum += elist[size_t(i)];
  }
  int last = nint;
  int64_t neval = 21 * int64_t(nint);
  double dres = std::fabs(result), errbnd = std::max(epsabs, epsrel * dres);
  if (abserr <= 100 * epmach * resabs && abserr > errbnd) ier = 2;
  if (nint != 1) {
    for (int i = 1; i <= npts; ++i) {
      int jlow = i + 1, ind1 = iord[size_t(i)], k = i;
      for (int j = jlow; j <= nint; ++j) {
        int ind2 = iord[size_t(j)];
        if (elist[size_t(ind1)] > elist[size_t(ind2)]) continue;
        ind1 = ind2;
        k = j;
      }
      if (ind1 == iord[size_t(i)]) continue;
      iord[size_t(k)] = iord[size_t(i)];
      iord[size_t(i)] = ind1;
    }
    if (limit < npts2) ier = 1;
  }
  auto done = [&]() {
    R.value = result * sign; R.abserr = abserr; R.ier = ier; R.last = last; R.neval = neval;
    return R;
  };
  if (ier != 0 || abserr <= errbnd) return done();
  rlist2[1] = result;
  int maxerr = iord[1];
  double errmax = elist[size_t(maxerr)], area = result, erlarg = errsum, ertest = errbnd, correc = 0;
  int nrmax = 1, nres = 0, numrl2 = 1, ktmin = 0, levmax = 1, iroff1 = 0, iroff2 = 0, iroff3 = 0, ierro = 0;
  bool extrap = false, noext = false;
  abserr = oflow;
  int ksgn = dres >= (1 - 50 * epmach) * resabs ? 1 : -1;
  bool sum_result = false;
  for (last = npts2; last <= limit; ++last) {
    int levcur = level[size_t(maxerr)] + 1;
    double a1x = alist[size_t(maxerr)], b1 = 0.5 * (alist[size_t(maxerr)] + blist[size_t(maxerr)]), a2 = b1, b2 = blist[size_t(maxerr)];
    double erlast = errmax, area1, error1, defab1, area2, error2, defab2, ra;
    qk21(f, a1x, b1, area1, error1, ra, defab1);
    qk21(f, a2, b2, area2, error2, ra, defab2);
    neval += 42;
    double area12 = area1 + area2, erro12 = error1 + error2;
    errsum = errsum + erro12 - errmax;
    area = area + area12 - rlist[size_t(maxerr)];
    if (!(defab1 == error1 || defab2 == error2)) {
      if (!(std::fabs(rlist[size_t(maxerr)] - area12) > 1e-5 * std::fabs(area12) || erro12 < 0.99 * errmax)) {
        if (extrap) ++iroff2;
        else ++iroff1;
      }
      if (last > 10 && erro12 > errmax) ++iroff3;
    }
    level[size_t(maxerr)] = levcur;
    level[size_t(last)] = levcur;
    rlist[size_t(maxerr)] = area1;
    rlist[size_t(last)] = area2;
    errbnd = std::max(epsabs, epsrel * std::fabs(area));
    if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
    if (iroff2 >= 5) ierro = 3;
    if (last == limit) ier = 1;
    if (std::max(std::fabs(a1x), std::fabs(b2)) <= (1 + 100 * epmach) * (std::fabs(a2) + 1000 * uflow)) ier = 4;
    if (error2 > error1) {
      alist[size_t(maxerr)] = a2; alist[size_t(last)] = a1x; blist[size_t(last)] = b1;
      rlist[size_t(maxerr)] = area2; rlist[size_t(last)] = area1;
      elist[size_t(maxerr)] = error2; elist[size_t(last)] = error1;
    } else {
      alist[size_t(last)] = a2; blist[size_t(maxerr)] = b1; blist[size_t(last)] = b2;
      elist[size_t(maxerr)] = error1; elist[size_t(last)] = error2;
    }
    qpsrt(limit, last, maxerr, errmax, elist.data(), iord.data(), nrmax);
    if (errsum <= errbnd) { sum_result = true; break; }
    if (ier != 0) break;
    if (noext) continue;
    erlarg -= erlast;
    if (levcur + 1 <= levmax) erlarg += erro12;
    if (!extrap) {
      if (level[size_t(maxerr)] + 1 <= levmax) continue;
      extrap = true;
      nrmax = 2;
    }
    bool skip = false;
    if (!(ierro == 3 || erlarg <= ertest)) {
      int id = nrmax, jupbnd = last;
      if (last > (2 + limit / 2)) jupbnd = limit + 3 - last;
      for (int k = id; k <= jupbnd; ++k) {
        maxerr = iord[size_t(nrmax)];
        errmax = elist[size_t(maxerr)];
        if (level[size_t(maxerr)] + 1 <= levmax) { skip = true; break; }
        ++nrmax;
      }
    }
    if (skip) continue;
    ++numrl2;
    rlist2[numrl2] = area;
    bool stop = false;
    if (numrl2 > 2) {
      double reseps, abseps;
      qelg(numrl2, rlist2, reseps, abseps, res3la, nres);
      ++ktmin;
      if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
      if (abseps < abserr) {
        ktmin = 0;
        abserr = abseps;
        result = reseps;
        correc = erlarg;
        ertest = std::max(epsabs, epsrel * std::fabs(reseps));
        if (abserr < ertest) stop = true;
      }
      if (!stop) {
        if (numrl2 == 1) noext = true;
        if (ier >= 5) stop = true;
      }
    }
    if (stop) break;
    maxerr = iord[1];
    errmax = elist[size_t(maxerr)];
    nrmax = 1;
    extrap = false;
    ++levmax;
    erlarg = errsum;
  }
  if (last > limit) last = limit;
  if (!sum_result) {
    bool divergence_test = false;
    if (abserr == oflow) sum_result = true;
    else if (ier + ierro == 0) divergence_test = true;
    else {
      if (ierro == 3) abserr += correc;
      if (ier == 0) ier = 3;
      if (result != 0 && area != 0) {
        if (abserr / std::fabs(result) > errsum / std::fabs(area)) sum_result = true;
        else divergence_test = true;
      } else if (abserr > errsum) sum_result = true;
      else if (area == 0) { }
      else divergence_test = true;
    }
    if (divergence_test && !(ksgn == -1 && std::max(std::fabs(result), std::fabs(area)) <= resabs * 0.01))
      if (0.01 > (result / area) || (result / area) > 100 || errsum > std::fabs(area)) ier = 6;
  }
  if (sum_result) {
    result = 0;
    for (int k = 1; k <= last; ++k) result += rlist[size_t(k)];
    abserr = errsum;
  }
  if (ier > 2) --ier;
  return done();
}

}  // namespace

QuadResult quad_integrate(const std::function<double(double)>& f, double a, double b, double epsabs, double epsrel, int limit, const std::vector<double>& points) {
  bool flip = b < a;
  double lo = std::min(a, b), hi = std::max(a, b);
  QuadResult r;
  if (!points.empty()) {
    if (std::isinf(lo) || std::isinf(hi)) fail("quad: points need finite limits");
    std::vector<double> pts;
    for (double p : points) if (p > lo && p < hi) pts.push_back(p);
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    r = qagp(f, lo, hi, pts, epsabs, epsrel, limit);
  } else if (std::isfinite(lo) && std::isfinite(hi)) {
    r = qags_core([&](double x1, double x2, double& res, double& err, double& rabs, double& rasc) { qk21(f, x1, x2, res, err, rabs, rasc); }, lo, hi,
                  epsabs, epsrel, limit, 21, 1);
  } else {
    int inf;
    double bound;
    if (hi == INF && lo != -INF) { inf = 1; bound = lo; }
    else if (hi == INF && lo == -INF) { inf = 2; bound = 0; }
    else { inf = -1; bound = hi; }
    r = qags_core([&](double x1, double x2, double& res, double& err, double& rabs, double& rasc) { qk15i(f, bound, inf, x1, x2, res, err, rabs, rasc); },
                  0.0, 1.0, epsabs, epsrel, limit, 15, inf == 2 ? 2 : 1);
    if (inf == -1) r.value = r.value;  // DQK15I already maps (-inf, b] with dinf = -1
  }
  if (flip) r.value = -r.value;
  return r;
}

std::string quad_message(int ier, int limit) {
  switch (ier) {
    case 0: return "";
    case 1: return "The maximum number of subdivisions (" + std::to_string(limit) + ") has been achieved.";
    case 2: return "The occurrence of roundoff error is detected, which prevents ";
    case 3: return "Extremely bad integrand behavior occurs at some points of the";
    case 4: return "The algorithm does not converge.  Roundoff error is detected";
    case 5: return "The integral is probably divergent, or slowly convergent.";
    case 6: return "The input is invalid.";
  }
  return "Unknown error.";
}

}  // namespace lmnum::opt
