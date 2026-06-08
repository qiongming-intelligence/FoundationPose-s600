#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "foundationpose_s600/core/tensor.hpp"

namespace foundationpose_s600 {

enum class BpuMemoryLocation {
  kHost,
  kBpu,
  kCpu,
};

enum class BpuCacheMode {
  kNonCacheable,
  kCacheable,
  kCoherent,
};

struct BpuAllocationOptions {
  BpuMemoryLocation location{BpuMemoryLocation::kBpu};
  BpuCacheMode cache_mode{BpuCacheMode::kNonCacheable};
  bool shared_iova{true};
  std::string label;
};

class BpuBuffer {
 public:
  BpuBuffer();
  ~BpuBuffer();

  BpuBuffer(const BpuBuffer&) = delete;
  BpuBuffer& operator=(const BpuBuffer&) = delete;
  BpuBuffer(BpuBuffer&&) noexcept;
  BpuBuffer& operator=(BpuBuffer&&) noexcept;

  void Reset();
  void CleanCache() const;
  void InvalidateCache() const;

  [[nodiscard]] std::uint8_t* CpuData() const;
  [[nodiscard]] std::uint64_t BpuAddress() const;
  [[nodiscard]] std::uint64_t PhysicalAddress() const;
  [[nodiscard]] std::uint64_t DeviceIova(std::uint32_t core_index) const;
  [[nodiscard]] std::uint64_t HostIova(std::uint32_t core_index) const;
  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] bool Empty() const;
  [[nodiscard]] bool BpuBacked() const;
  [[nodiscard]] bool Cacheable() const;
  [[nodiscard]] BpuMemoryLocation Location() const;

 private:
  friend class BpuAllocator;

  struct Impl;

  explicit BpuBuffer(std::unique_ptr<Impl> impl);

  void Release() noexcept;

  std::unique_ptr<Impl> impl_;
};

struct BpuTensorBuffer {
  TensorInfo info;
  BpuBuffer buffer;
};

[[nodiscard]] std::uint64_t TensorStorageBytes(const TensorInfo& info);

class BpuAllocator {
 public:
  [[nodiscard]] BpuBuffer Allocate(std::size_t bytes) const;
  [[nodiscard]] BpuBuffer Allocate(std::size_t bytes, BpuAllocationOptions options) const;
  [[nodiscard]] BpuTensorBuffer AllocateTensor(const TensorInfo& info, BpuAllocationOptions options = {}) const;
};

}  // namespace foundationpose_s600
