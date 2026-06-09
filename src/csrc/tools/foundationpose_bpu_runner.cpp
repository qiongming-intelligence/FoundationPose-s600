#include "foundationpose_s600/bpu/bpu_model.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace foundationpose_s600 {
namespace {

struct LoadedModel {
  std::string key;
  std::string hbm;
  std::string partition;
  BpuModel model;
  std::vector<BpuTensorBuffer> inputs;
  std::vector<BpuTensorBuffer> outputs;
};

struct RunnerOptions {
  std::string core_id{"0"};
  std::uint64_t backend_mask{0};
};

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          out += ' ';
        } else {
          out += ch;
        }
        break;
    }
  }
  return out;
}

std::string JsonString(const std::string& value) { return "\"" + JsonEscape(value) + "\""; }

std::string JsonUnescape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\' || i + 1 >= value.size()) {
      out += value[i];
      continue;
    }
    const char next = value[++i];
    switch (next) {
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case '\\':
      case '"':
      case '/':
        out += next;
        break;
      default:
        out += next;
        break;
    }
  }
  return out;
}

std::string ExtractString(const std::string& json, const std::string& key, const std::string& default_value = {}) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    return default_value;
  }
  return JsonUnescape(match[1].str());
}

std::map<std::string, std::string> ExtractObjectStrings(const std::string& json, const std::string& key) {
  std::map<std::string, std::string> out;
  const std::regex object_start("\\\"" + key + "\\\"\\s*:\\s*\\{");
  std::smatch match;
  if (!std::regex_search(json, match, object_start)) {
    return out;
  }
  std::size_t begin = static_cast<std::size_t>(match.position()) + static_cast<std::size_t>(match.length());
  std::size_t depth = 1;
  std::size_t end = begin;
  for (; end < json.size(); ++end) {
    if (json[end] == '{') {
      ++depth;
    } else if (json[end] == '}') {
      --depth;
      if (depth == 0) {
        break;
      }
    }
  }
  if (end >= json.size()) {
    throw std::runtime_error("unterminated JSON object for key: " + key);
  }
  const std::string body = json.substr(begin, end - begin);
  const std::regex pair_pattern("\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), pair_pattern); it != std::sregex_iterator(); ++it) {
    out[JsonUnescape((*it)[1].str())] = JsonUnescape((*it)[2].str());
  }
  return out;
}

std::string DTypeName(TensorDataType dtype) {
  switch (dtype) {
    case TensorDataType::kInt4:
      return "int4";
    case TensorDataType::kUint4:
      return "uint4";
    case TensorDataType::kInt8:
      return "int8";
    case TensorDataType::kUint8:
      return "uint8";
    case TensorDataType::kFloat16:
      return "float16";
    case TensorDataType::kInt16:
      return "int16";
    case TensorDataType::kUint16:
      return "uint16";
    case TensorDataType::kFloat32:
      return "float32";
    case TensorDataType::kInt32:
      return "int32";
    case TensorDataType::kUint32:
      return "uint32";
    case TensorDataType::kFloat64:
      return "float64";
    case TensorDataType::kInt64:
      return "int64";
    case TensorDataType::kUint64:
      return "uint64";
    case TensorDataType::kBool8:
      return "bool8";
    case TensorDataType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::uint64_t DTypeBytes(TensorDataType dtype) {
  switch (dtype) {
    case TensorDataType::kInt4:
    case TensorDataType::kUint4:
    case TensorDataType::kInt8:
    case TensorDataType::kUint8:
    case TensorDataType::kBool8:
      return 1;
    case TensorDataType::kFloat16:
    case TensorDataType::kInt16:
    case TensorDataType::kUint16:
      return 2;
    case TensorDataType::kFloat32:
    case TensorDataType::kInt32:
    case TensorDataType::kUint32:
      return 4;
    case TensorDataType::kFloat64:
    case TensorDataType::kInt64:
    case TensorDataType::kUint64:
      return 8;
    case TensorDataType::kUnknown:
      return 0;
  }
  return 0;
}

std::uint64_t CheckedMul(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs != 0 && lhs > UINT64_MAX / rhs) {
    throw std::overflow_error("tensor byte size overflow");
  }
  return lhs * rhs;
}

std::uint64_t TensorValidBytes(const TensorInfo& info) {
  const auto element_bytes = DTypeBytes(info.dtype);
  if (element_bytes == 0 || info.shape.dims.empty()) {
    return TensorStorageBytes(info);
  }
  std::uint64_t elements = 1;
  for (int dim : info.shape.dims) {
    if (dim < 0) {
      return TensorStorageBytes(info);
    }
    elements = CheckedMul(elements, static_cast<std::uint64_t>(dim));
  }
  return CheckedMul(elements, element_bytes);
}

std::uint64_t BackendMaskFromCoreId(const std::string& core_id) {
  // Match hrt_model_exec semantics: 0 means any BPU core, 1 means core 0,
  // 2 means core 1, etc. Comma-separated values select multiple cores.
  std::uint64_t mask = 0;
  std::stringstream ss(core_id);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch) == 0; }));
    token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; }).base(), token.end());
    if (token.empty()) {
      continue;
    }
    const auto value = std::stoul(token);
    if (value == 0) {
      return 0;
    }
    if (value > 64) {
      throw std::runtime_error("--core-id value out of range: " + token);
    }
    mask |= (std::uint64_t{1} << (value - 1));
  }
  return mask;
}

std::string ShapeJson(const TensorShape& shape) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < shape.dims.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << shape.dims[i];
  }
  oss << ']';
  return oss.str();
}

std::string TensorInfoJson(const TensorInfo& info) {
  std::ostringstream oss;
  oss << '{'
      << "\"name\":" << JsonString(info.name) << ','
      << "\"shape\":" << ShapeJson(info.shape) << ','
      << "\"dtype\":" << JsonString(DTypeName(info.dtype)) << ','
      << "\"valid_bytes\":" << TensorValidBytes(info) << ','
      << "\"aligned_bytes\":" << info.byte_size << ','
      << "\"stride\":[";
  for (std::size_t i = 0; i < info.stride.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << info.stride[i];
  }
  oss << "]}";
  return oss.str();
}

std::string TensorInfoArrayJson(const std::vector<TensorInfo>& infos) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < infos.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << TensorInfoJson(infos[i]);
  }
  oss << ']';
  return oss.str();
}

std::string ModelInfoJson(const LoadedModel& loaded) {
  std::ostringstream oss;
  oss << '{'
      << "\"key\":" << JsonString(loaded.key) << ','
      << "\"hbm\":" << JsonString(loaded.hbm) << ','
      << "\"partition\":" << JsonString(loaded.partition) << ','
      << "\"model_name\":" << JsonString(loaded.model.Name()) << ','
      << "\"compile_bpu_core_num\":" << loaded.model.CompileBpuCoreNum() << ','
      << "\"inputs\":" << TensorInfoArrayJson(loaded.model.Inputs()) << ','
      << "\"outputs\":" << TensorInfoArrayJson(loaded.model.Outputs()) << '}';
  return oss.str();
}

std::string SafeFileName(std::string name) {
  for (char& ch : name) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '.' || ch == '_' || ch == '-')) {
      ch = '_';
    }
  }
  return name.empty() ? std::string{"tensor"} : name;
}

std::uint64_t FileSize(const fs::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open input tensor: " + path.string());
  }
  const auto size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to stat input tensor: " + path.string());
  }
  return static_cast<std::uint64_t>(size);
}

void ReadTensorFile(const fs::path& path, BpuTensorBuffer& tensor) {
  const auto size = FileSize(path);
  const auto valid_bytes = TensorValidBytes(tensor.info);
  if (size != valid_bytes && size != tensor.buffer.Size()) {
    std::ostringstream oss;
    oss << "input tensor size mismatch for " << tensor.info.name << ": " << path << " has " << size
        << " bytes, expected " << valid_bytes << " valid bytes or " << tensor.buffer.Size() << " aligned bytes";
    throw std::runtime_error(oss.str());
  }
  if (size > tensor.buffer.Size()) {
    throw std::runtime_error("input tensor too large for BPU buffer: " + tensor.info.name);
  }
  auto* data = tensor.buffer.CpuData();
  if (data == nullptr) {
    throw std::runtime_error("input tensor has no CPU address: " + tensor.info.name);
  }
  std::fill_n(data, tensor.buffer.Size(), std::uint8_t{0});
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
  if (!input) {
    throw std::runtime_error("failed to read input tensor: " + path.string());
  }
  tensor.buffer.CleanCache();
}

void WriteCompactTensorRecursive(std::ofstream& output,
                                 const std::uint8_t* base,
                                 const TensorInfo& info,
                                 std::size_t dim,
                                 std::uint64_t offset_bytes,
                                 std::uint64_t element_bytes) {
  if (dim + 1 == info.shape.dims.size()) {
    const auto count = static_cast<std::uint64_t>(info.shape.dims[dim]);
    const auto bytes = CheckedMul(count, element_bytes);
    output.write(reinterpret_cast<const char*>(base + offset_bytes), static_cast<std::streamsize>(bytes));
    return;
  }
  const auto count = static_cast<std::uint64_t>(info.shape.dims[dim]);
  const auto stride = static_cast<std::uint64_t>(info.stride[dim]);
  for (std::uint64_t i = 0; i < count; ++i) {
    WriteCompactTensorRecursive(output, base, info, dim + 1, offset_bytes + CheckedMul(i, stride), element_bytes);
  }
}

bool HasUsableStride(const TensorInfo& info) {
  if (info.shape.dims.empty() || info.stride.size() != info.shape.dims.size()) {
    return false;
  }
  for (std::size_t i = 0; i < info.shape.dims.size(); ++i) {
    if (info.shape.dims[i] < 0 || info.stride[i] < 0) {
      return false;
    }
  }
  return true;
}

void WriteTensorFile(const fs::path& path, const BpuTensorBuffer& tensor) {
  const auto bytes = TensorValidBytes(tensor.info);
  if (bytes > tensor.buffer.Size()) {
    throw std::runtime_error("output tensor valid bytes exceed aligned buffer: " + tensor.info.name);
  }
  const auto* data = tensor.buffer.CpuData();
  if (data == nullptr) {
    throw std::runtime_error("output tensor has no CPU address: " + tensor.info.name);
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open output tensor: " + path.string());
  }
  const auto element_bytes = DTypeBytes(tensor.info.dtype);
  if (HasUsableStride(tensor.info) && element_bytes != 0) {
    WriteCompactTensorRecursive(output, data, tensor.info, 0, 0, element_bytes);
  } else {
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  }
  if (!output) {
    throw std::runtime_error("failed to write output tensor: " + path.string());
  }
}

LoadedModel LoadModelSpec(const std::string& spec) {
  const auto eq = spec.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
    throw std::runtime_error("--model must be key=/path/to/model.hbm[:partition], got: " + spec);
  }
  LoadedModel loaded;
  loaded.key = spec.substr(0, eq);
  std::string rhs = spec.substr(eq + 1);
  const auto colon = rhs.rfind(':');
  if (colon != std::string::npos) {
    loaded.hbm = rhs.substr(0, colon);
    loaded.partition = rhs.substr(colon + 1);
  } else {
    loaded.hbm = rhs;
  }
  if (loaded.hbm.empty()) {
    throw std::runtime_error("empty HBM path in --model: " + spec);
  }
  std::cerr << "[foundationpose-bpu-runner] loading " << loaded.key << " from " << loaded.hbm << std::endl;
  loaded.model.Load(loaded.hbm);
  loaded.inputs = loaded.model.AllocateInputs();
  loaded.outputs = loaded.model.AllocateOutputs();
  std::cerr << "[foundationpose-bpu-runner] loaded " << loaded.key << " model_name=" << loaded.model.Name()
            << " inputs=" << loaded.inputs.size() << " outputs=" << loaded.outputs.size() << std::endl;
  return loaded;
}

std::string AllModelInfoJson(const std::map<std::string, LoadedModel>& models) {
  std::ostringstream oss;
  oss << "{\"ok\":true,\"models\":{";
  bool first = true;
  for (const auto& item : models) {
    if (!first) {
      oss << ',';
    }
    first = false;
    oss << JsonString(item.first) << ':' << ModelInfoJson(item.second);
  }
  oss << "}}";
  return oss.str();
}

std::string RunInfer(LoadedModel& loaded, const std::string& line, const RunnerOptions& options) {
  const auto input_paths = ExtractObjectStrings(line, "inputs");
  const std::string output_dir_str = ExtractString(line, "output_dir");
  if (output_dir_str.empty()) {
    throw std::runtime_error("infer command missing output_dir");
  }
  const fs::path output_dir{output_dir_str};
  fs::create_directories(output_dir);

  for (auto& tensor : loaded.inputs) {
    auto it = input_paths.find(tensor.info.name);
    if (it == input_paths.end()) {
      throw std::runtime_error("infer command missing input: " + tensor.info.name);
    }
    ReadTensorFile(fs::path{it->second}, tensor);
  }

  loaded.model.Infer(loaded.inputs, loaded.outputs, 0, options.backend_mask);

  std::ostringstream outputs_obj;
  std::ostringstream outputs_order;
  outputs_obj << '{';
  outputs_order << '[';
  for (std::size_t i = 0; i < loaded.outputs.size(); ++i) {
    const auto& tensor = loaded.outputs[i];
    const fs::path output_path = output_dir / (SafeFileName(tensor.info.name) + ".bin");
    WriteTensorFile(output_path, tensor);
    if (i != 0) {
      outputs_obj << ',';
      outputs_order << ',';
    }
    outputs_obj << JsonString(tensor.info.name) << ':' << JsonString(output_path.string());
    outputs_order << '{'
                  << "\"name\":" << JsonString(tensor.info.name) << ','
                  << "\"path\":" << JsonString(output_path.string()) << ','
                  << "\"shape\":" << ShapeJson(tensor.info.shape) << ','
                  << "\"dtype\":" << JsonString(DTypeName(tensor.info.dtype)) << ','
                  << "\"valid_bytes\":" << TensorValidBytes(tensor.info) << '}';
  }
  outputs_obj << '}';
  outputs_order << ']';

  std::ostringstream response;
  response << "{\"ok\":true,\"model\":" << JsonString(loaded.key) << ",\"outputs\":" << outputs_obj.str()
           << ",\"output_order\":" << outputs_order.str() << '}';
  return response.str();
}

std::string HandleCommand(std::map<std::string, LoadedModel>& models, const std::string& line, bool& shutdown, const RunnerOptions& options) {
  const std::string cmd = ExtractString(line, "cmd");
  if (cmd.empty()) {
    throw std::runtime_error("command missing cmd");
  }
  if (cmd == "shutdown") {
    shutdown = true;
    return "{\"ok\":true,\"shutdown\":true}";
  }
  if (cmd == "model_info") {
    const std::string key = ExtractString(line, "model");
    if (key.empty()) {
      return AllModelInfoJson(models);
    }
    auto it = models.find(key);
    if (it == models.end()) {
      throw std::runtime_error("unknown model key: " + key);
    }
    return "{\"ok\":true,\"model\":" + ModelInfoJson(it->second) + "}";
  }
  if (cmd == "infer") {
    const std::string key = ExtractString(line, "model");
    if (key.empty()) {
      throw std::runtime_error("infer command missing model");
    }
    auto it = models.find(key);
    if (it == models.end()) {
      throw std::runtime_error("unknown model key: " + key);
    }
    return RunInfer(it->second, line, options);
  }
  throw std::runtime_error("unsupported command: " + cmd);
}

void PrintUsage(const char* argv0) {
  std::cout << "usage: " << argv0 << " --model key=/path/model.hbm[:partition] [--model key2=/path/model2.hbm[:partition]] [--model-info]\n"
            << "\n"
            << "Persistent FoundationPose S600 BPU JSON-lines runner. Commands are read from stdin; responses are written to stdout.\n"
            << "\n"
            << "Commands:\n"
            << "  {\"cmd\":\"model_info\"}\n"
            << "  {\"cmd\":\"infer\",\"model\":\"refine\",\"inputs\":{\"A\":\"/tmp/A.bin\",\"B\":\"/tmp/B.bin\"},\"output_dir\":\"/tmp/out\"}\n"
            << "  {\"cmd\":\"shutdown\"}\n";
}

}  // namespace
}  // namespace foundationpose_s600

int main(int argc, char** argv) {
  using namespace foundationpose_s600;
  try {
    std::vector<std::string> model_specs;
    bool print_model_info = false;
    RunnerOptions options;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        PrintUsage(argv[0]);
        return 0;
      }
      if (arg == "--model") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--model requires an argument");
        }
        model_specs.push_back(argv[++i]);
        continue;
      }
      if (arg == "--model-info") {
        print_model_info = true;
        continue;
      }
      if (arg == "--protocol") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--protocol requires an argument");
        }
        const std::string protocol = argv[++i];
        if (protocol != "jsonl") {
          throw std::runtime_error("only --protocol jsonl is supported");
        }
        continue;
      }
      if (arg == "--core-id") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--core-id requires an argument");
        }
        options.core_id = argv[++i];
        options.backend_mask = BackendMaskFromCoreId(options.core_id);
        std::cerr << "[foundationpose-bpu-runner] --core-id " << options.core_id
                  << (options.backend_mask == 0 ? " (BPU_CORE_ANY)" : " (explicit BPU backend mask)") << std::endl;
        continue;
      }
      throw std::runtime_error("unknown argument: " + arg);
    }
    if (model_specs.empty()) {
      throw std::runtime_error("at least one --model is required");
    }

    std::map<std::string, LoadedModel> models;
    for (const auto& spec : model_specs) {
      LoadedModel loaded = LoadModelSpec(spec);
      const std::string key = loaded.key;
      if (models.find(key) != models.end()) {
        throw std::runtime_error("duplicate model key: " + key);
      }
      models.emplace(key, std::move(loaded));
    }

    if (print_model_info) {
      std::cout << AllModelInfoJson(models) << std::endl;
      return 0;
    }

    std::string line;
    bool shutdown = false;
    while (!shutdown && std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      try {
        std::cout << HandleCommand(models, line, shutdown, options) << std::endl;
      } catch (const std::exception& exc) {
        std::cout << "{\"ok\":false,\"error\":" << JsonString(exc.what()) << "}" << std::endl;
      }
      std::cout.flush();
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "foundationpose_bpu_runner: " << exc.what() << std::endl;
    return 1;
  }
}
