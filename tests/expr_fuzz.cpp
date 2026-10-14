// Deterministic mutation fuzzing of the expression parser under ASan+UBSan. The parser may only accept or throw
// ParseFailure; every accepted AST must respect the spec caps and the declared names/dims (the exact invariants the
// Python builder re-checks).
#include "expr_parser.hpp"
#include <cstdio>
#include <fstream>
#include <random>
using lmcpp::Json;
static size_t count_nodes(const Json& n, size_t depth, size_t& deepest, const std::set<std::string>& names, const std::map<std::string,size_t>& dims) {
    deepest = std::max(deepest, depth);
    size_t total = 1;
    const std::string kind = n.at("t");
    if (kind == "var") { if (!names.count(n.at("n").get<std::string>()) || dims.count(n.at("n").get<std::string>())) std::abort(); }
    else if (kind == "idx") { auto it = dims.find(n.at("n").get<std::string>()); if (it == dims.end() || n.at("i").get<size_t>() >= it->second) std::abort(); }
    else if (kind == "ifelse") { total += count_nodes(n.at("c"), depth + 1, deepest, names, dims) + count_nodes(n.at("a"), depth + 1, deepest, names, dims) + count_nodes(n.at("b"), depth + 1, deepest, names, dims); }
    else if (n.contains("a")) for (const auto& child : n.at("a")) total += count_nodes(child, depth + 1, deepest, names, dims);
    return total;
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream shared(argv[1]), extra(argv[2]);
    Json corpus1, corpus2; shared >> corpus1; extra >> corpus2;
    std::vector<std::string> seeds;
    for (const Json* corpus : {&corpus1, &corpus2}) for (const char* group : {"accept", "reject"}) for (const auto& f : (*corpus)[group]) seeds.push_back(f["expression"].get<std::string>());
    const std::vector<std::string> alphabet = {"x", "y", "v", "[", "]", "0", "1", "9", "63", "(", ")", "+", "-", "*", "/", "^", ",", " ", "e", ".", "<", "&", "|", "ifelse(", "exp(", "sum(", "pi", "\xd0\xb5", "\xff", "\t", "\n", "%", "'", "#", "[[", "]]", "1e", "--"};
    std::mt19937_64 rng(2026);
    unsigned accepted = 0, rejected = 0;
    const std::set<std::string> names = {"x", "y", "v"};
    const std::map<std::string,size_t> dims_options[] = {{}, {{"v", 2}}, {{"v", 4}, {"x", 1}}};
    for (unsigned i = 0; i < 15000; ++i) {
        std::string text = seeds[rng() % seeds.size()];
        for (int edits = 1 + int(rng() % 4); edits > 0; --edits) {
            switch (rng() % 4) {
            case 0: if (!text.empty()) text.erase(rng() % text.size(), 1 + rng() % 3); break;
            case 1: text.insert(rng() % (text.size() + 1), alphabet[rng() % alphabet.size()]); break;
            case 2: if (!text.empty()) text.replace(rng() % text.size(), 1, alphabet[rng() % alphabet.size()]); break;
            default: text += alphabet[rng() % alphabet.size()]; break;
            }
        }
        if (rng() % 50 == 0) text = std::string(rng() % 4000, 'x');
        const auto& dims = dims_options[rng() % 3];
        try {
            pycli_expr::Parser parser(text, names, rng() % 2 == 0, dims);
            Json ast = parser.parse();
            size_t deepest = 0;
            size_t nodes = count_nodes(ast, 0, deepest, names, dims);
            if (nodes > 512 || text.size() > 2048) std::abort();
            ++accepted;
        } catch (const pycli_expr::ParseFailure&) { ++rejected; }
    }
    std::fprintf(stderr, "expr fuzz: %u accepted, %u rejected\n", accepted, rejected);
    return accepted >= 150 ? 0 : 3;  // the mutator must keep producing valid expressions, not just garbage
}
