// Expression engine shared by optimisation, ODEs, frames and Vec.apply. The grammar is pycli's expr-spec.json:
// numbers, names, + - * / ^ (right-assoc, binds tighter than unary minus), comparisons, & |, ifelse(cond, a, b), the
// constants pi and e, and the spec's functions. Names bind to scalars or to vectors; a vector is read as name[i]
// (literal index) or passed whole to a reducer (sum, mean, prod, min/max with one argument). Expressions compile once to
// a flat node array evaluated recursively; evaluation allocates nothing.
#pragma once
#include <map>
#include <string>
#include <vector>

namespace lmnum {

struct ExprProgram {
  enum Code : unsigned char { NUM, VAR, IDX, VREDUCE, NEG, ADD, SUB, MUL, DIV, POW, LT, LE, GT, GE, EQ, NE, AND, OR, IFELSE, CALL1, CALL2 };
  struct Ins { Code code; int fn = 0; int slot = 0; int index = 0; double value = 0; };
  std::vector<Ins> code;
  std::string text;
};

// Compiles `text` with the given scalar names (slots 0..n-1 of the evaluation array) and vector names (slots of the vector
// table). Unknown names, bad syntax and wrong arities are errors that quote the text and the position.
class ExprCompiler {
 public:
  ExprCompiler(std::vector<std::string> scalars, std::vector<std::string> vectors = {});
  ExprProgram compile(const std::string& text) const;
  int scalar_slot(const std::string& name) const;  // -1 when absent
 private:
  std::vector<std::string> scalars_, vectors_;
};

// Evaluates a compiled program. `vars` holds the scalar slots; `vecs` the vector slots (pointer + length).
struct VecRef { const double* data; size_t n; };
double expr_eval(const ExprProgram& p, const double* vars, const VecRef* vecs = nullptr);

// Names used by an expression (scalars and vectors), in first-use order; used to infer parameters.
std::vector<std::string> expr_names(const std::string& text);

}  // namespace lmnum
