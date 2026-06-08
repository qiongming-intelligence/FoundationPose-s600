#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace foundationpose_s600 {

enum class TensorDataType {
  kUnknown,
  kInt4,
  kUint4,
  kInt8,
  kUint8,
  kFloat16,
  kInt16,
  kUint16,
  kFloat32,
  kInt32,
  kUint32,
  kFloat64,
  kInt64,
  kUint64,
  kBool8,
};

struct TensorShape {
  std::vector<int> dims;
};

struct TensorInfo {
  std::string name;
  TensorShape shape;
  TensorDataType dtype{TensorDataType::kUnknown};
  std::uint64_t byte_size{0};
  std::vector<std::int64_t> stride;
};

}  // namespace foundationpose_s600
