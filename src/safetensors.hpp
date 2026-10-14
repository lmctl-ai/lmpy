#pragma once
#include <lmcpp/lmcpp.hpp>
#include <cstdint>
#include <string>
#include <map>
#include <vector>
namespace pycli {
struct TensorData { std::string dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
struct TensorBundle { std::map<std::string, TensorData> tensors; std::map<std::string, std::string> metadata; };
struct TensorRef { std::string path, key; };
TensorRef parse_ref(const std::string& ref);
TensorData read_tensor(const TensorRef& ref);
TensorBundle read_bundle(const std::string& path);
std::string write_bundle(const std::string& path, const TensorBundle& bundle);
std::string write_tensor(const std::string& path, const TensorData& tensor);
lmcpp::Json descriptor(const TensorRef& ref, const TensorData& tensor);
}
