// Initial value problems: pycli `ode_solve` / `ode_solve_to` (scipy.integrate.solve_ivp) on expression right-hand sides.
// The solvers are a port of scipy's (ode_solvers.cpp); this file is solve_ivp's driver (t_eval sampling, events located
// with brentq on the dense output, terminal events) plus pycli's contract on top: the default 101-point output grid,
// endpoints added to an explicit t_eval, evaluation/step caps, non-finite values stop the run, and pycli's result fields
// and statuses.
#include "expr.hpp"
#include "host.hpp"
#include "ode_solvers.hpp"
#include <algorithm>
#include <cctype>
#include <set>

namespace lmnum {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
const std::string Rt = "Res", Vt = "Vec", St = "StrVec", OSt = "OdeSol";

// A budget or validity stop inside the integration (pycli SolverStop): the run ends with this status and a partial result.
struct Stop { std::string status, message; };

const char* MSG_END = "The solver successfully reached the end of the integration interval.";
const char* MSG_EVENT = "A termination event occurred.";

struct Problem {
  std::vector<std::string> states;
  std::vector<ExprProgram> rhs, events;
  std::vector<double> event_dir;
  std::vector<double> event_max;       // occurrences after which the run stops (scipy max_events)
  std::vector<bool> event_terminal;    // user asked for termination (status OK) rather than the cap (EVENT_LIMIT)
  std::vector<std::string> event_text;
  ExprBindings bind;
  std::vector<VecRef> bind_refs;
  VectorXd y0;
  double t0 = 0, t1 = 0;
  std::string method = "RK45";
  ode::Settings settings;
  std::vector<double> grid;
  std::vector<double> endpoints_added;
  int64_t max_evals = 200000, max_steps = 10000;
};

bool valid_name(const std::string& s) {
  if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
  for (char c : s) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
  return true;
}

Problem prepare(const std::vector<std::string>& rhs, const std::vector<std::string>& states, const std::vector<double>& y0, double t0, double t1, const Opts& o, const char* op) {
  const std::string OP = op;
  Problem p;
  size_t n = y0.size();
  if (n == 0) fail(OP + ": y0 is empty");
  if (states.size() != n) fail(OP + ": " + std::to_string(states.size()) + " state names for " + std::to_string(n) + " initial values");
  if (rhs.size() != n) fail(OP + ": " + std::to_string(rhs.size()) + " right-hand sides for " + std::to_string(n) + " states (one expression per state)");
  for (double v : y0) if (!std::isfinite(v)) fail(OP + ": y0 must be finite");
  if (!std::isfinite(t0) || !std::isfinite(t1)) fail(OP + ": t0 and t1 must be finite");
  if (t0 == t1) fail(OP + ": t0 and t1 must differ");
  std::set<std::string> seen = {"t", "y", "pi", "e"};
  for (auto& s : states) {
    if (!valid_name(s)) fail(OP + ": state name '" + s + "' is not an identifier");
    if (!seen.insert(s).second) fail(OP + ": state name '" + s + "' is duplicated or reserved (t, y, pi, e)");
  }
  p.states = states;
  p.y0 = to_eigen(y0);
  p.t0 = t0; p.t1 = t1;

  p.method = o.str("method", "RK45");
  static const std::set<std::string> methods = {"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"};
  if (!methods.count(p.method)) fail(OP + ": method must be RK45, RK23, DOP853, Radau, BDF or LSODA; got '" + p.method + "'");
  double rtol = o.num("rtol", 1e-6);
  if (!(rtol > 0) || !std::isfinite(rtol)) fail(OP + ": rtol must be a positive finite number");
  p.settings.rtol = rtol;
  p.settings.atol = VectorXd::Constant(Eigen::Index(n), 1e-9);
  if (const Item* a = o.find("atol")) {
    if (a->kind == Item::VEC) {
      if (a->vec.size() != n) fail(OP + ": atol has " + std::to_string(a->vec.size()) + " values for " + std::to_string(n) + " states");
      p.settings.atol = to_eigen(a->vec);
    } else if (a->kind == Item::NUM || a->kind == Item::INT) p.settings.atol.setConstant(a->kind == Item::NUM ? a->f : double(a->i));
    else fail(OP + ": atol must be a float or a Vec with one value per state");
    for (Eigen::Index i = 0; i < p.settings.atol.size(); ++i)
      if (!(p.settings.atol(i) >= 0) || !std::isfinite(p.settings.atol(i))) fail(OP + ": atol must be finite and >= 0");
  }
  p.settings.max_step = o.num("max_step", INFINITY);
  if (!(p.settings.max_step > 0)) fail(OP + ": max_step must be positive");
  p.settings.first_step = o.num("first_step", 0);
  if (o.has("first_step") && !(p.settings.first_step > 0 && p.settings.first_step <= std::fabs(t1 - t0)))
    fail(OP + ": first_step must be positive and at most |t1 - t0|");
  p.max_evals = o.integer("max_evals", 200000);
  p.max_steps = o.integer("max_steps", 10000);
  if (p.max_evals < 1 || p.max_steps < 1) fail(OP + ": max_evals and max_steps must be >= 1");

  // output grid (pycli: explicit t_eval with the endpoints added, else max_output_points uniform samples)
  double sign = t1 > t0 ? 1 : -1;
  if (const auto* te = o.vec("t_eval")) {
    if (o.has("max_output_points")) fail(OP + ": give t_eval or max_output_points, not both");
    if (te->empty()) fail(OP + ": t_eval is empty");
    p.grid = *te;
    for (size_t i = 0; i < p.grid.size(); ++i) {
      double g = p.grid[i];
      if (!std::isfinite(g) || sign * (g - t0) < 0 || sign * (t1 - g) < 0 || (i && sign * (g - p.grid[i - 1]) <= 0))
        fail(OP + ": t_eval must be strictly ordered from t0 towards t1 and lie within [t0, t1]");
    }
    if (p.grid.front() != t0) { p.grid.insert(p.grid.begin(), t0); p.endpoints_added.push_back(t0); }
    if (p.grid.back() != t1) { p.grid.push_back(t1); p.endpoints_added.push_back(t1); }
  } else {
    int64_t count = o.integer("max_output_points", 101);
    if (count < 2) fail(OP + ": max_output_points must be >= 2");
    if (count > 10000000) fail(OP + ": max_output_points must be <= 1e7");
    for (int64_t i = 0; i < count; ++i)
      p.grid.push_back(double(static_cast<long double>(t0) + (static_cast<long double>(t1) - t0) * static_cast<long double>(i) / static_cast<long double>(count - 1)));
    p.grid.front() = t0;
    p.grid.back() = t1;
    p.grid.erase(std::unique(p.grid.begin(), p.grid.end()), p.grid.end());
  }

  // expressions: scalars = states, t, $scalars; vectors = y, $vectors
  p.bind = expr_bindings(o);
  std::vector<std::string> scalars = states;
  scalars.push_back("t");
  for (auto& s : p.bind.scalar_names) {
    if (seen.count(s)) fail(OP + ": binding $" + s + " collides with a state name or a reserved name (t, y, pi, e)");
    seen.insert(s);
    scalars.push_back(s);
  }
  std::vector<std::string> vectors = {"y"};
  for (auto& s : p.bind.vector_names) {
    if (seen.count(s)) fail(OP + ": binding $" + s + " collides with a state name or a reserved name (t, y, pi, e)");
    seen.insert(s);
    vectors.push_back(s);
  }
  p.bind_refs = p.bind.refs();
  ExprCompiler comp(scalars, vectors);
  for (auto& r : rhs) p.rhs.push_back(comp.compile(r));

  const std::vector<std::string> none;
  const auto* ev = o.strs("events");
  const auto& events = ev ? *ev : none;
  const auto* dir = o.vec("direction");
  const auto* term = o.vec("terminal");
  int64_t max_events = o.integer("max_events", 32);
  if (max_events < 1) fail(OP + ": max_events must be >= 1");
  if (dir && dir->size() != events.size()) fail(OP + ": direction needs one value per event");
  if (term && term->size() != events.size()) fail(OP + ": terminal needs one value per event");
  for (size_t i = 0; i < events.size(); ++i) {
    p.events.push_back(comp.compile(events[i]));
    p.event_text.push_back(events[i]);
    double d = dir ? (*dir)[i] : 0;
    if (d != -1 && d != 0 && d != 1) fail(OP + ": event direction must be -1, 0 or 1");
    p.event_dir.push_back(d);
    double k = term ? (*term)[i] : 0;
    if (!(k >= 0) || k != std::floor(k)) fail(OP + ": terminal must be 0 (not terminal) or a positive whole number of occurrences (1 = stop at the first)");
    p.event_terminal.push_back(k > 0);
    p.event_max.push_back(k > 0 ? k : double(max_events));
  }
  o.finish(op);
  return p;
}

// One integration (solve_ivp with pycli's wrappers). Fills the result; keeps the dense pieces when `dense` is set.
struct Run {
  const Problem& P;
  size_t n;
  std::vector<double> vars;
  std::vector<VecRef> refs;
  int64_t evals = 0, rhs_calls = 0, event_calls = 0, nonfinite = 0, steps = 0;
  double track_time;
  VectorXd track_state;

  explicit Run(const Problem& p) : P(p), n(size_t(p.y0.size())), track_time(p.t0), track_state(p.y0) {
    vars.assign(n + 1 + p.bind.scalar_names.size(), 0.0);
    for (size_t i = 0; i < p.bind.scalar_values.size(); ++i) vars[n + 1 + i] = p.bind.scalar_values[i];
    refs.push_back(VecRef{nullptr, n});
    for (auto& r : p.bind_refs) refs.push_back(r);
  }
  void tick() {
    if (evals >= P.max_evals) throw Stop{"EVAL_LIMIT", "expression evaluation cap of " + std::to_string(P.max_evals) + " reached"};
    ++evals;
  }
  void bind(double t, const VectorXd& y) {
    if (!y.allFinite()) { tick(); ++nonfinite; throw Stop{"NON_FINITE", "non-finite trial state"}; }
    for (size_t i = 0; i < n; ++i) vars[i] = y(Eigen::Index(i));
    vars[n] = t;
    refs[0].data = y.data();
  }
  double scalar(const ExprProgram& prog) {
    tick();
    double v = expr_eval(prog, vars.data(), refs.data());
    if (!std::isfinite(v)) { ++nonfinite; throw Stop{"NON_FINITE", "non-finite RHS/event value"}; }
    return v;
  }
  void rhs(double t, const VectorXd& y, VectorXd& out) {
    ++rhs_calls;
    bind(t, y);
    out.resize(Eigen::Index(n));
    for (size_t i = 0; i < n; ++i) out(Eigen::Index(i)) = scalar(P.rhs[i]);
  }
  double event(size_t k, double t, const VectorXd& y) {
    ++event_calls;
    bind(t, y);
    double v = scalar(P.events[k]);
    // a zero exactly at the initial time has no preceding crossing (pycli)
    return (t == P.t0 && v == 0) ? NAN : v;
  }

  Res* go(bool dense, ode::Solution* solution) {
    std::vector<double> ts;
    std::vector<VectorXd> ys;
    size_t ne = P.events.size();
    std::vector<std::vector<double>> t_events(ne);
    std::vector<std::vector<VectorXd>> y_events(ne);
    std::vector<double> event_count(ne, 0.0);
    std::string status = "OK", message, termination = "end";
    bool partial = false;
    double last_time = P.t0;
    VectorXd last_state = P.y0;
    int scipy_status = 0;
    std::unique_ptr<ode::Solver> solver;
    long nfev = 0, njev = 0, nlu = 0;
    double direction = P.t1 > P.t0 ? 1 : -1;
    std::vector<double> ti = {P.t0};
    try {
      ode::Rhs f = [this](double t, const VectorXd& y, VectorXd& out) { rhs(t, y, out); };
      solver = ode::make_solver(P.method, f, P.t0, P.y0, P.t1, P.settings);
      std::vector<double> g(ne);
      for (size_t k = 0; k < ne; ++k) g[k] = event(k, P.t0, P.y0);
      size_t gi = 0;
      int st = -2;  // None
      while (st == -2) {
        if (steps >= P.max_steps) throw Stop{"STEP_LIMIT", "accepted-step cap reached"};
        std::string msg = solver->step();
        nfev = solver->nfev; njev = solver->njev; nlu = solver->nlu;
        if (!solver->y.allFinite()) { ++nonfinite; throw Stop{"NON_FINITE", "non-finite accepted state"}; }
        if (solver->status != "failed") { ++steps; track_time = solver->t; track_state = solver->y; }
        if (solver->status == "finished") st = 0;
        else if (solver->status == "failed") { st = -1; message = msg; break; }
        double t_old = solver->t_old, t = solver->t;
        VectorXd y = solver->y;
        ode::DensePtr sol;
        if (dense) { sol = solver->dense_output(); nfev = solver->nfev; }
        auto need_sol = [&]() { if (!sol) { sol = solver->dense_output(); nfev = solver->nfev; } };
        if (ne) {
          std::vector<double> g_new(ne);
          for (size_t k = 0; k < ne; ++k) g_new[k] = event(k, t, y);
          std::vector<size_t> active;
          for (size_t k = 0; k < ne; ++k) {  // find_active_events
            bool up = g[k] <= 0 && g_new[k] >= 0, down = g[k] >= 0 && g_new[k] <= 0;
            double d = P.event_dir[k];
            if ((up && d > 0) || (down && d < 0) || ((up || down) && d == 0)) active.push_back(k);
          }
          if (!active.empty()) {
            need_sol();
            for (size_t k : active) event_count[k] += 1;
            std::vector<double> roots;
            for (size_t k : active) {  // solve_event_equation
              const double EPS4 = 4 * std::numeric_limits<double>::epsilon();
              bool conv = true;
              VectorXd ys_;
              double r = ode::brentq([&](double tt) { sol->eval(tt, ys_); return event(k, tt, ys_); }, t_old, t, EPS4, EPS4, 100, conv);
              roots.push_back(r);
            }
            bool terminate = false;
            for (size_t k : active) if (event_count[k] >= P.event_max[k]) terminate = true;
            if (terminate) {  // handle_events: keep the roots up to the first event that reached its count
              std::vector<size_t> order(active.size());
              for (size_t i = 0; i < order.size(); ++i) order[i] = i;
              std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return t > t_old ? roots[a] < roots[b] : -roots[a] < -roots[b]; });
              std::vector<size_t> act2; std::vector<double> roots2;
              for (size_t i : order) { act2.push_back(active[i]); roots2.push_back(roots[i]); if (event_count[active[i]] >= P.event_max[active[i]]) break; }
              active = act2; roots = roots2;
            }
            for (size_t i = 0; i < active.size(); ++i) {
              t_events[active[i]].push_back(roots[i]);
              y_events[active[i]].push_back((*sol)(roots[i]));
            }
            if (terminate) { st = 1; t = roots.back(); y = (*sol)(t); }
          }
          g = g_new;
        }
        // t_eval sampling: grid points up to and including t along the integration direction
        size_t gi_new = gi;
        while (gi_new < P.grid.size() && direction * (P.grid[gi_new] - t) <= 0) ++gi_new;
        if (gi_new > gi) {
          need_sol();
          for (size_t k = gi; k < gi_new; ++k) { ts.push_back(P.grid[k]); ys.push_back((*sol)(P.grid[k])); }
          gi = gi_new;
        }
        if (dense) { solution->pieces.push_back(sol); ti.push_back(t); }
      }
      scipy_status = st;
      nfev = solver->nfev; njev = solver->njev; nlu = solver->nlu;
      last_time = track_time; last_state = track_state;
      if (st < 0) {
        status = (message.find("step size") != std::string::npos || message.find("Step size") != std::string::npos) ? "STEP_SIZE_TOO_SMALL" : "NO_CONVERGENCE";
        termination = "failed"; partial = true;
      } else message = st == 0 ? MSG_END : MSG_EVENT;
      if (st == 1) {
        // the first terminal root along the integration direction (the accepted step may extend beyond it)
        bool found = false; size_t idx = 0; double best = 0;
        for (size_t k = 0; k < ne; ++k) {
          if (t_events[k].empty() || event_count[k] < P.event_max[k]) continue;
          double key = (t_events[k].back() - P.t0) * direction;
          if (!found || key < best) { found = true; best = key; idx = k; }
        }
        if (found) {
          last_time = t_events[idx].back(); last_state = y_events[idx].back();
          termination = "event";
          if (!P.event_terminal[idx]) { status = "EVENT_LIMIT"; termination = "failed"; partial = true;
            message = "event " + std::to_string(idx) + " reached its cap of " + jnum(P.event_max[idx]) + " occurrences (max_events)"; }
        }
      }
    } catch (const Stop& s) {
      status = s.status; message = s.message; partial = true; termination = "failed";
      last_time = track_time; last_state = track_state;
      if (solver) { nfev = solver->nfev; njev = solver->njev; nlu = solver->nlu; }
    }
    if (ts.empty() || ts.back() != last_time) { ts.push_back(last_time); ys.push_back(last_state); }
    if (dense && solution) {
      solution->ts = ti;
      solution->alt_segment = solver ? solver->alt_segment() : false;
      if (solution->pieces.size() + 1 != solution->ts.size()) solution->ts.resize(solution->pieces.size() + 1);
    }

    auto* r = make<Res>();
    r->str("status", status).str("message", message).str("termination", termination).str("method", P.method);
    r->num("final_time", last_time).vec("final_state", last_state).strs("state_names", P.states);
    r->vec("t", ts);
    MatrixXd Y(Eigen::Index(ys.size()), Eigen::Index(n));
    for (size_t i = 0; i < ys.size(); ++i) Y.row(Eigen::Index(i)) = ys[i].transpose();
    r->mat("y", Y);
    std::vector<double> event_t, event_index, counts;
    std::vector<VectorXd> event_y;
    for (size_t k = 0; k < ne; ++k) {
      counts.push_back(double(t_events[k].size()));
      for (size_t i = 0; i < t_events[k].size(); ++i) { event_t.push_back(t_events[k][i]); event_y.push_back(y_events[k][i]); event_index.push_back(double(k)); }
    }
    MatrixXd EY(Eigen::Index(event_y.size()), Eigen::Index(n));
    for (size_t i = 0; i < event_y.size(); ++i) EY.row(Eigen::Index(i)) = event_y[i].transpose();
    r->vec("event_counts", counts).vec("event_t", event_t).mat("event_y", EY).vec("event_index", event_index);
    r->integer("nfev", nfev).integer("njev", njev).integer("nlu", nlu).integer("steps", steps);
    r->integer("rhs_calls", rhs_calls).integer("event_calls", event_calls).integer("evals", evals).integer("nonfinite_evals", nonfinite);
    r->flag("success", scipy_status >= 0 && status == "OK").flag("partial_trajectory", partial);
    r->integer("effective_output_points", int64_t(P.grid.size())).vec("endpoints_added", P.endpoints_added);
    return r;
  }
};

// Dense solution object (scipy solve_ivp(dense_output=True).sol) with the run's result.
struct OdeSol : Obj {
  static constexpr const char* TYPE = "OdeSol";
  Res* res = nullptr;
  ode::Solution sol;
  double t_min = 0, t_max = 0;
  size_t n = 0;
  VectorXd at(double t) const {
    if (sol.pieces.empty()) fail("OdeSol.eval: the integration made no step");
    if (!(t >= t_min && t <= t_max)) fail("OdeSol.eval: t = " + jnum(t) + " is outside the integrated range [" + jnum(t_min) + ", " + jnum(t_max) + "]");
    VectorXd out;
    sol.eval(t, out);
    return out;
  }
};

const std::vector<std::string>& strs_of(const Value& v) { return obj<StrVec>(v).v; }

Res* run_ode(Args& a, const Opts& o, const char* op) {
  Problem p = prepare(strs_of(a[0]), strs_of(a[1]), obj<Vec>(a[2]).v, a[3].f, a[4].f, o, op);
  Run run(p);
  return run.go(false, nullptr);
}

OdeSol* run_sol(Args& a, const Opts& o) {
  Problem p = prepare(strs_of(a[0]), strs_of(a[1]), obj<Vec>(a[2]).v, a[3].f, a[4].f, o, "ode_sol");
  Run run(p);
  auto* s = make<OdeSol>();
  s->res = run.go(true, &s->sol);
  s->n = size_t(p.y0.size());
  if (!s->sol.ts.empty()) {
    s->t_min = std::min(s->sol.ts.front(), s->sol.ts.back());
    s->t_max = std::max(s->sol.ts.front(), s->sol.ts.back());
  }
  return s;
}

}  // namespace

void register_ode(Api& api) {
  api.section("Ordinary differential equations",
              "Initial value problems dy/dt = f(t, y) with expression right-hand sides (pycli `ode_solve`, `ode_solve_to`; "
              "scipy.integrate.solve_ivp). `rhs[i]` is the expression for d(state[i])/dt in the state names, `t`, `y[i]` and "
              "`$`-bound constants (`opts().set(\"$beta\", 0.3)`). Methods RK45 (default), RK23, DOP853, Radau, BDF are ports of "
              "scipy's solvers (same step control, dense output and finite-difference Jacobian); LSODA is an automatic switch "
              "from RK45 to BDF when Hairer's stiffness test fires, not ODEPACK. Options (pycli names and defaults): `method`, "
              "`rtol` (1e-6), `atol` (1e-9, float or one per state), `t_eval` (ordered times; t0/t1 are added when missing) or "
              "`max_output_points` (101 uniform samples), `max_step`, `first_step`, `events` (StrVec of expressions whose zeros "
              "are located by brentq on the dense output), `direction` (Vec of -1/0/1), `terminal` (Vec: 0 = record only, k = stop "
              "at the k-th occurrence), `max_events` (32: a non-terminal event stops the run with EVENT_LIMIT at that count), "
              "`max_evals` (200000 expression evaluations), `max_steps` (10000 accepted steps). Result statuses: OK, "
              "STEP_SIZE_TOO_SMALL, NO_CONVERGENCE, EVAL_LIMIT, STEP_LIMIT, EVENT_LIMIT, NON_FINITE (a NaN/inf RHS or event "
              "value stops the run); a non-OK run keeps the samples computed so far and ends at the last accepted state.");
  const std::string resdoc =
      "result: status, message, termination (end/event/failed), method, final_time, final_state, state_names, t (Vec), "
      "y (Mat: one row per time, one column per state), event_counts, event_t, event_y (Mat), event_index (grouped by event), "
      "nfev, njev, nlu, steps, rhs_calls, event_calls, evals, nonfinite_evals, success, partial_trajectory, "
      "effective_output_points, endpoints_added";
  std::vector<std::string> base = {St, St, Vt, "float", "float"};
  std::vector<std::string> with = {St, St, Vt, "float", "float", "Opts"};
  api.fn("ode", base, Rt, "solve an initial value problem ode(rhs, state, y0, t0, t1) with RK45 at rtol 1e-6, atol 1e-9 on 101 points; " + resdoc,
         [](Args& a) { return O(run_ode(a, no_opts(), "ode")); });
  api.fn("ode", with, Rt, "solve an initial value problem with options (method, rtol, atol, t_eval, events, ... and $ constants); same result fields",
         [](Args& a) { return O(run_ode(a, obj<Opts>(a[5]), "ode")); });

  api.type(OSt, "Dense ODE solution (solve_ivp dense_output): the piecewise interpolant of every accepted step plus the run's result.");
  api.fn("ode_sol", base, OSt, "like ode, keeping the dense output for evaluation at any time inside the integrated range",
         [](Args& a) { return O(run_sol(a, no_opts())); });
  api.fn("ode_sol", with, OSt, "like ode with options, keeping the dense output (DOP853 then spends 3 extra RHS calls per step, as in scipy)",
         [](Args& a) { return O(run_sol(a, obj<Opts>(a[5]))); });
  api.method(OSt, "res", {}, Rt, "the run's result (the fields of `ode`)", [](void* s, Args&) { return O(static_cast<OdeSol*>(s)->res); });
  api.method(OSt, "eval", {"float"}, Vt, "state vector at time t (t must lie in the integrated range)", [](void* s, Args& a) {
    return vecv(static_cast<OdeSol*>(s)->at(a[0].f)); });
  api.method(OSt, "eval", {Vt}, "Mat", "states at several times: one row per time, one column per state", [](void* s, Args& a) {
    auto* o = static_cast<OdeSol*>(s);
    const auto& ts = obj<Vec>(a[0]).v;
    MatrixXd m(Eigen::Index(ts.size()), Eigen::Index(o->n));
    for (size_t i = 0; i < ts.size(); ++i) m.row(Eigen::Index(i)) = o->at(ts[i]).transpose();
    return matv(std::move(m)); });
  api.method(OSt, "t_min", {}, "float", "start of the integrated range", [](void* s, Args&) { return F(static_cast<OdeSol*>(s)->t_min); });
  api.method(OSt, "t_max", {}, "float", "end of the integrated range (the terminal event time when one stopped the run)", [](void* s, Args&) { return F(static_cast<OdeSol*>(s)->t_max); });
}

}  // namespace lmnum
