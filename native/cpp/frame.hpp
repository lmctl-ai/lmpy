// Frame: a small typed column table (float columns with NaN as missing, and text columns).
#pragma once
#include "host.hpp"

namespace lmnum {

struct Column {
  std::string name;
  bool numeric = true;
  std::vector<double> num;         // numeric == true (NaN marks a missing value)
  std::vector<std::string> text;   // numeric == false
  size_t size() const { return numeric ? num.size() : text.size(); }
  std::string cell_text(size_t i) const;  // numbers as shortest round-trip text, NaN as ""
};

struct Frame : Obj {
  static constexpr const char* TYPE = "Frame";
  std::vector<Column> cols;
  size_t nrows() const { return cols.empty() ? 0 : cols[0].size(); }
  const Column* find(const std::string& name) const;
  const Column& get(const std::string& name) const;  // error lists the columns
  const Column& num_col(const std::string& name) const;  // must be numeric
  void add(Column c);  // replaces a column of the same name; length must match
  std::string names_text() const;
};

// CSV reading/writing (RFC 4180 quoting; a column is numeric when every non-empty cell parses as a finite or NA number).
Frame* read_csv_file(const std::string& path, const Opts& o);
Frame* parse_csv_text(const std::string& text, const Opts& o);
std::string frame_to_csv(const Frame& f);
std::string frame_to_json(const Frame& f);  // {"columns": [...], "rows": n, "data": {name: [...]}}

// Rows selected by index, in the given order (used by filter, sort, head ...).
Frame* frame_take(const Frame& f, const std::vector<size_t>& rows);

// Atomic file write: temporary file in the destination directory, then rename.
void write_file_atomic(const std::string& path, const std::string& bytes);
std::string read_file_text(const std::string& path);

}  // namespace lmnum
