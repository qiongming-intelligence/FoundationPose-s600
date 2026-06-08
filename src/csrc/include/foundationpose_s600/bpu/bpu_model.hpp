#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "foundationpose_s600/bpu/bpu_allocator.hpp"
#include "foundationpose_s600/core/tensor.hpp"

namespace foundationpose_s600 {

class BpuModel {
 public:
  BpuModel();
  explicit BpuModel(std::string model_path);
  ~BpuModel();

  BpuModel(const BpuModel&) = delete;
  BpuModel& operator=(const BpuModel&) = delete;
  BpuModel(BpuModel&&) noexcept;
  BpuModel& operator=(BpuModel&&) noexcept;

  void Load(std::string model_path);
  void Reset();
  void Infer(const std::vector<BpuTensorBuffer>& inputs,
             std::vector<BpuTensorBuffer>& outputs,
             std::int32_t timeout_ms = 0,
             std::uint64_t backend_mask = 0) const;

  [[nodiscard]] bool Loaded() const;
  [[nodiscard]] const std::string& Path() const;
  [[nodiscard]] const std::string& Name() const;
  [[nodiscard]] const std::vector<TensorInfo>& Inputs() const;
  [[nodiscard]] const std::vector<TensorInfo>& Outputs() const;
  [[nodiscard]] std::vector<BpuTensorBuffer> AllocateInputs(BpuAllocationOptions options = {}) const;
  [[nodiscard]] std::vector<BpuTensorBuffer> AllocateOutputs(BpuAllocationOptions options = {}) const;
  [[nodiscard]] int CompileBpuCoreNum() const;

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
  std::string path_;
  std::string name_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
  int compile_bpu_core_num_{0};
  bool loaded_{false};
};

}  // namespace foundationpose_s600
