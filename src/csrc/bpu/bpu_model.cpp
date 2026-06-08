#include "foundationpose_s600/bpu/bpu_model.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)
#include <hobot/dnn/hb_dnn.h>
#include <hobot/dnn/hb_dnn_status.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_status.h>
#endif

namespace foundationpose_s600 {
namespace {

std::string Basename(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)

void CheckDnn(int32_t code, const std::string& action) {
  if (code != 0) {
    throw std::runtime_error(action + " failed: " + hbDNNGetErrorDesc(code));
  }
}

void CheckUcp(int32_t code, const std::string& action) {
  if (code != 0) {
    throw std::runtime_error(action + " failed: " + hbUCPGetErrorDesc(code));
  }
}

TensorDataType ConvertDnnType(int32_t type) {
  switch (type) {
    case HB_DNN_TENSOR_TYPE_S4:
      return TensorDataType::kInt4;
    case HB_DNN_TENSOR_TYPE_U4:
      return TensorDataType::kUint4;
    case HB_DNN_TENSOR_TYPE_S8:
      return TensorDataType::kInt8;
    case HB_DNN_TENSOR_TYPE_U8:
      return TensorDataType::kUint8;
    case HB_DNN_TENSOR_TYPE_F16:
      return TensorDataType::kFloat16;
    case HB_DNN_TENSOR_TYPE_S16:
      return TensorDataType::kInt16;
    case HB_DNN_TENSOR_TYPE_U16:
      return TensorDataType::kUint16;
    case HB_DNN_TENSOR_TYPE_F32:
      return TensorDataType::kFloat32;
    case HB_DNN_TENSOR_TYPE_S32:
      return TensorDataType::kInt32;
    case HB_DNN_TENSOR_TYPE_U32:
      return TensorDataType::kUint32;
    case HB_DNN_TENSOR_TYPE_F64:
      return TensorDataType::kFloat64;
    case HB_DNN_TENSOR_TYPE_S64:
      return TensorDataType::kInt64;
    case HB_DNN_TENSOR_TYPE_U64:
      return TensorDataType::kUint64;
    case HB_DNN_TENSOR_TYPE_BOOL8:
      return TensorDataType::kBool8;
    default:
      return TensorDataType::kUnknown;
  }
}

TensorInfo TensorInfoFromDnn(const char* name, const hbDNNTensorProperties& properties) {
  TensorInfo info;
  info.name = name == nullptr ? std::string{} : std::string{name};
  info.dtype = ConvertDnnType(properties.tensorType);
  info.byte_size = properties.alignedByteSize < 0 ? 0 : static_cast<std::uint64_t>(properties.alignedByteSize);
  for (int32_t i = 0; i < properties.validShape.numDimensions; ++i) {
    info.shape.dims.push_back(properties.validShape.dimensionSize[i]);
    info.stride.push_back(properties.stride[i]);
  }
  return info;
}

std::int64_t AlignStride(std::int64_t value) { return (value + 63) & ~std::int64_t{63}; }

std::int64_t DnnElementBytes(int32_t tensor_type) {
  switch (ConvertDnnType(tensor_type)) {
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
      return 1;
  }
  return 1;
}

void NormalizeDynamicStride(hbDNNTensorProperties& properties) {
  const int32_t dim_count = properties.validShape.numDimensions;
  for (int32_t i = dim_count - 1; i >= 0; --i) {
    if (properties.stride[i] != -1) {
      continue;
    }
    if (i + 1 >= dim_count) {
      properties.stride[i] = DnnElementBytes(properties.tensorType);
      continue;
    }
    properties.stride[i] = AlignStride(properties.stride[i + 1] * properties.validShape.dimensionSize[i + 1]);
  }
}

void ValidateTensorBuffers(const std::vector<BpuTensorBuffer>& buffers,
                           const std::vector<TensorInfo>& expected,
                           const std::string& kind) {
  if (buffers.size() != expected.size()) {
    throw std::runtime_error(kind + " tensor count mismatch");
  }
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const auto required_bytes = TensorStorageBytes(expected[i]);
    if (required_bytes != 0 && buffers[i].buffer.Size() < required_bytes) {
      throw std::runtime_error(kind + " tensor buffer too small: " + expected[i].name);
    }
    if (buffers[i].buffer.CpuData() == nullptr) {
      throw std::runtime_error(kind + " tensor buffer has no CPU address: " + expected[i].name);
    }
  }
}

hbDNNTensor MakeDnnTensor(const BpuTensorBuffer& buffer, const hbDNNTensorProperties& properties) {
  hbDNNTensor tensor{};
  tensor.properties = properties;
  tensor.sysMem.virAddr = buffer.buffer.CpuData();
  tensor.sysMem.phyAddr = buffer.buffer.PhysicalAddress();
  tensor.sysMem.memSize = buffer.buffer.Size();
  return tensor;
}

std::uint64_t BackendMaskForCompileCoreNum(int compile_core_num) {
  switch (compile_core_num) {
    case 2:
      return HB_UCP_BPU_CORE_0 | HB_UCP_BPU_CORE_1;
    case 3:
      return HB_UCP_BPU_CORE_0 | HB_UCP_BPU_CORE_1 | HB_UCP_BPU_CORE_2;
    case 4:
      return HB_UCP_BPU_CORE_0 | HB_UCP_BPU_CORE_1 | HB_UCP_BPU_CORE_2 | HB_UCP_BPU_CORE_3;
    case 1:
    default:
      return HB_UCP_BPU_CORE_ANY;
  }
}

#endif

}  // namespace

struct BpuModel::Impl {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)
  hbDNNPackedHandle_t packed_handle{nullptr};
  hbDNNHandle_t model_handle{nullptr};
  std::vector<hbDNNTensorProperties> input_properties;
  std::vector<hbDNNTensorProperties> output_properties;
#endif
};

BpuModel::BpuModel() : impl_(std::make_unique<Impl>()) {}

BpuModel::BpuModel(std::string model_path) : BpuModel() { Load(std::move(model_path)); }

BpuModel::~BpuModel() { Reset(); }

BpuModel::BpuModel(BpuModel&&) noexcept = default;

BpuModel& BpuModel::operator=(BpuModel&& other) noexcept {
  if (this != &other) {
    Reset();
    impl_ = std::move(other.impl_);
    path_ = std::move(other.path_);
    name_ = std::move(other.name_);
    inputs_ = std::move(other.inputs_);
    outputs_ = std::move(other.outputs_);
    compile_bpu_core_num_ = other.compile_bpu_core_num_;
    loaded_ = other.loaded_;
    other.compile_bpu_core_num_ = 0;
    other.loaded_ = false;
  }
  return *this;
}

void BpuModel::Load(std::string model_path) {
  Reset();
  path_ = std::move(model_path);
  name_ = Basename(path_);

  if (path_.empty()) {
    throw std::runtime_error("BPU model path is empty");
  }
  if (!std::filesystem::exists(path_)) {
    throw std::runtime_error("BPU model file does not exist: " + path_);
  }

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)
  const char* files[] = {path_.c_str()};
  CheckDnn(hbDNNInitializeFromFiles(&impl_->packed_handle, files, 1), "hbDNNInitializeFromFiles");

  char const** model_names = nullptr;
  int32_t model_count = 0;
  CheckDnn(hbDNNGetModelNameList(&model_names, &model_count, impl_->packed_handle), "hbDNNGetModelNameList");
  if (model_count <= 0 || model_names == nullptr || model_names[0] == nullptr) {
    throw std::runtime_error("HBM file contains no DNN models: " + path_);
  }

  name_ = model_names[0];
  CheckDnn(hbDNNGetModelHandle(&impl_->model_handle, impl_->packed_handle, name_.c_str()), "hbDNNGetModelHandle");

  int32_t input_count = 0;
  CheckDnn(hbDNNGetInputCount(&input_count, impl_->model_handle), "hbDNNGetInputCount");
  for (int32_t i = 0; i < input_count; ++i) {
    char const* input_name = nullptr;
    hbDNNTensorProperties properties{};
    CheckDnn(hbDNNGetInputName(&input_name, impl_->model_handle, i), "hbDNNGetInputName");
    CheckDnn(hbDNNGetInputTensorProperties(&properties, impl_->model_handle, i), "hbDNNGetInputTensorProperties");
    NormalizeDynamicStride(properties);
    impl_->input_properties.push_back(properties);
    inputs_.push_back(TensorInfoFromDnn(input_name, properties));
  }

  int32_t output_count = 0;
  CheckDnn(hbDNNGetOutputCount(&output_count, impl_->model_handle), "hbDNNGetOutputCount");
  for (int32_t i = 0; i < output_count; ++i) {
    char const* output_name = nullptr;
    hbDNNTensorProperties properties{};
    CheckDnn(hbDNNGetOutputName(&output_name, impl_->model_handle, i), "hbDNNGetOutputName");
    CheckDnn(hbDNNGetOutputTensorProperties(&properties, impl_->model_handle, i), "hbDNNGetOutputTensorProperties");
    impl_->output_properties.push_back(properties);
    outputs_.push_back(TensorInfoFromDnn(output_name, properties));
  }

  CheckDnn(hbDNNGetCompileBpuCoreNum(&compile_bpu_core_num_, impl_->model_handle), "hbDNNGetCompileBpuCoreNum");
#else
  compile_bpu_core_num_ = 0;
#endif

  loaded_ = true;
}

void BpuModel::Reset() {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)
  if (impl_ && impl_->packed_handle != nullptr) {
    hbDNNRelease(impl_->packed_handle);
    impl_->packed_handle = nullptr;
    impl_->model_handle = nullptr;
  }
  if (impl_) {
    impl_->input_properties.clear();
    impl_->output_properties.clear();
  }
#endif
  inputs_.clear();
  outputs_.clear();
  compile_bpu_core_num_ = 0;
  path_.clear();
  name_.clear();
  loaded_ = false;
}

void BpuModel::Infer(const std::vector<BpuTensorBuffer>& inputs,
                     std::vector<BpuTensorBuffer>& outputs,
                     std::int32_t timeout_ms,
                     std::uint64_t backend_mask) const {
  if (!loaded_) {
    throw std::runtime_error("BPU model is not loaded");
  }

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_DNN)
  ValidateTensorBuffers(inputs, inputs_, "input");
  ValidateTensorBuffers(outputs, outputs_, "output");

  std::vector<hbDNNTensor> input_tensors;
  input_tensors.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    input_tensors.push_back(MakeDnnTensor(inputs[i], impl_->input_properties[i]));
    inputs[i].buffer.CleanCache();
  }

  std::vector<hbDNNTensor> output_tensors;
  output_tensors.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    output_tensors.push_back(MakeDnnTensor(outputs[i], impl_->output_properties[i]));
  }

  hbUCPTaskHandle_t task_handle{nullptr};
  CheckDnn(hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), impl_->model_handle), "hbDNNInferV2");

  hbUCPSchedParam sched_param;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
  sched_param.backend = backend_mask == 0 ? BackendMaskForCompileCoreNum(compile_bpu_core_num_) : backend_mask;

  try {
    CheckUcp(hbUCPSubmitTask(task_handle, &sched_param), "hbUCPSubmitTask");
    CheckUcp(hbUCPWaitTaskDone(task_handle, timeout_ms), "hbUCPWaitTaskDone");
    for (auto& output : outputs) {
      output.buffer.InvalidateCache();
    }
  } catch (...) {
    hbUCPReleaseTask(task_handle);
    throw;
  }

  CheckUcp(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask");
#else
  (void)inputs;
  (void)outputs;
  (void)timeout_ms;
  throw std::runtime_error("Hobot DNN runtime is not available");
#endif
}

bool BpuModel::Loaded() const { return loaded_; }

const std::string& BpuModel::Path() const { return path_; }

const std::string& BpuModel::Name() const { return name_; }

const std::vector<TensorInfo>& BpuModel::Inputs() const { return inputs_; }

const std::vector<TensorInfo>& BpuModel::Outputs() const { return outputs_; }

std::vector<BpuTensorBuffer> BpuModel::AllocateInputs(BpuAllocationOptions options) const {
  options.location = BpuMemoryLocation::kCpu;
  options.cache_mode = BpuCacheMode::kCacheable;
  BpuAllocator allocator;
  std::vector<BpuTensorBuffer> buffers;
  buffers.reserve(inputs_.size());
  for (const auto& input : inputs_) {
    buffers.push_back(allocator.AllocateTensor(input, options));
  }
  return buffers;
}

std::vector<BpuTensorBuffer> BpuModel::AllocateOutputs(BpuAllocationOptions options) const {
  options.location = BpuMemoryLocation::kCpu;
  options.cache_mode = BpuCacheMode::kCacheable;
  BpuAllocator allocator;
  std::vector<BpuTensorBuffer> buffers;
  buffers.reserve(outputs_.size());
  for (const auto& output : outputs_) {
    buffers.push_back(allocator.AllocateTensor(output, options));
  }
  return buffers;
}

int BpuModel::CompileBpuCoreNum() const { return compile_bpu_core_num_; }

}  // namespace foundationpose_s600
