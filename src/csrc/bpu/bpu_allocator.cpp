#include "foundationpose_s600/bpu/bpu_allocator.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
#include <hb_bpu_mem.h>
#endif

namespace foundationpose_s600 {
namespace {

std::uint32_t ToAllocFlag(BpuCacheMode cache_mode, bool shared_iova) {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  std::uint32_t flag = 0;
  switch (cache_mode) {
    case BpuCacheMode::kNonCacheable:
      flag = BPU_NON_CACHEABLE;
      break;
    case BpuCacheMode::kCacheable:
      flag = BPU_CACHEABLE;
      break;
    case BpuCacheMode::kCoherent:
      flag = BPU_CONHERENCE;
      break;
  }
  if (shared_iova) {
    flag |= BPU_SHARED;
  }
  return flag;
#else
  (void)cache_mode;
  (void)shared_iova;
  return 0;
#endif
}

std::uint64_t DTypeStorageBits(TensorDataType dtype) {
  switch (dtype) {
    case TensorDataType::kInt4:
    case TensorDataType::kUint4:
      return 4;
    case TensorDataType::kInt8:
    case TensorDataType::kUint8:
    case TensorDataType::kBool8:
      return 8;
    case TensorDataType::kFloat16:
    case TensorDataType::kInt16:
    case TensorDataType::kUint16:
      return 16;
    case TensorDataType::kFloat32:
    case TensorDataType::kInt32:
    case TensorDataType::kUint32:
      return 32;
    case TensorDataType::kFloat64:
    case TensorDataType::kInt64:
    case TensorDataType::kUint64:
      return 64;
    case TensorDataType::kUnknown:
      return 0;
  }
  return 0;
}

std::uint64_t CheckedMul(std::uint64_t lhs, std::uint64_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    throw std::overflow_error("tensor byte size overflow");
  }
  return lhs * rhs;
}

void* AllocateHost(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void* memory = std::calloc(1, bytes);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

struct HostDeleter {
  void operator()(std::uint8_t* ptr) const noexcept { std::free(ptr); }
};

}  // namespace

struct BpuBuffer::Impl {
  std::shared_ptr<std::uint8_t> host_data;
  std::uint64_t bpu_addr{0};
  std::size_t size{0};
  BpuMemoryLocation location{BpuMemoryLocation::kHost};
  bool cacheable{false};
  std::uint32_t alloc_flag{0};
  std::string label;
};

BpuBuffer::BpuBuffer() = default;

BpuBuffer::BpuBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

BpuBuffer::~BpuBuffer() { Release(); }

BpuBuffer::BpuBuffer(BpuBuffer&&) noexcept = default;

BpuBuffer& BpuBuffer::operator=(BpuBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void BpuBuffer::Release() noexcept {
  if (!impl_) {
    return;
  }
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  if (impl_->bpu_addr != 0) {
    switch (impl_->location) {
      case BpuMemoryLocation::kBpu:
        hb_bpu_mem_free(impl_->bpu_addr);
        break;
      case BpuMemoryLocation::kCpu:
        hb_bpu_cpumem_free(impl_->bpu_addr);
        break;
      case BpuMemoryLocation::kHost:
        break;
    }
  }
#endif
  impl_.reset();
}

void BpuBuffer::Reset() { Release(); }

void BpuBuffer::CleanCache() const {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  if (impl_ && impl_->bpu_addr != 0 && impl_->cacheable) {
    hb_bpu_mem_cache_flush(impl_->bpu_addr, static_cast<std::uint64_t>(impl_->size), BPU_MEM_CLEAN);
  }
#endif
}

void BpuBuffer::InvalidateCache() const {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  if (impl_ && impl_->bpu_addr != 0 && impl_->cacheable) {
    hb_bpu_mem_cache_flush(impl_->bpu_addr, static_cast<std::uint64_t>(impl_->size), BPU_MEM_INVALIDATE);
  }
#endif
}

std::uint8_t* BpuBuffer::CpuData() const {
  if (!impl_) {
    return nullptr;
  }
  if (impl_->location == BpuMemoryLocation::kHost) {
    return impl_->host_data.get();
  }
  if (impl_->bpu_addr == 0) {
    return nullptr;
  }
  return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(impl_->bpu_addr));
}

std::uint64_t BpuBuffer::BpuAddress() const { return impl_ == nullptr ? 0 : impl_->bpu_addr; }

std::uint64_t BpuBuffer::PhysicalAddress() const {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  return impl_ == nullptr || impl_->bpu_addr == 0 ? 0 : hb_bpu_mem_phyaddr(impl_->bpu_addr);
#else
  return 0;
#endif
}

std::uint64_t BpuBuffer::DeviceIova(std::uint32_t core_index) const {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  return impl_ == nullptr || impl_->bpu_addr == 0 ? 0 : hb_bpu_mem_device_iova(impl_->bpu_addr, core_index);
#else
  (void)core_index;
  return 0;
#endif
}

std::uint64_t BpuBuffer::HostIova(std::uint32_t core_index) const {
#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  return impl_ == nullptr || impl_->bpu_addr == 0 ? 0 : hb_bpu_mem_host_iova(impl_->bpu_addr, core_index);
#else
  (void)core_index;
  return 0;
#endif
}

std::size_t BpuBuffer::Size() const { return impl_ == nullptr ? 0 : impl_->size; }

bool BpuBuffer::Empty() const { return Size() == 0; }

bool BpuBuffer::BpuBacked() const { return impl_ != nullptr && impl_->location != BpuMemoryLocation::kHost; }

bool BpuBuffer::Cacheable() const { return impl_ != nullptr && impl_->cacheable; }

BpuMemoryLocation BpuBuffer::Location() const {
  return impl_ == nullptr ? BpuMemoryLocation::kHost : impl_->location;
}

std::uint64_t TensorStorageBytes(const TensorInfo& info) {
  if (info.byte_size != 0) {
    return info.byte_size;
  }

  if (!info.stride.empty() && !info.shape.dims.empty() && info.stride[0] > 0 && info.shape.dims[0] >= 0) {
    return CheckedMul(static_cast<std::uint64_t>(info.stride[0]), static_cast<std::uint64_t>(info.shape.dims[0]));
  }

  const std::uint64_t bits_per_element = DTypeStorageBits(info.dtype);
  if (bits_per_element == 0 || info.shape.dims.empty()) {
    return 0;
  }

  std::uint64_t element_count = 1;
  for (int dim : info.shape.dims) {
    if (dim < 0) {
      return 0;
    }
    element_count = CheckedMul(element_count, static_cast<std::uint64_t>(dim));
  }

  const std::uint64_t total_bits = CheckedMul(element_count, bits_per_element);
  return (total_bits + 7U) / 8U;
}

BpuBuffer BpuAllocator::Allocate(std::size_t bytes) const {
  return Allocate(bytes, BpuAllocationOptions{});
}

BpuBuffer BpuAllocator::Allocate(std::size_t bytes, BpuAllocationOptions options) const {
  if (bytes == 0) {
    return BpuBuffer{};
  }

  auto impl = std::make_unique<BpuBuffer::Impl>();
  impl->size = bytes;
  impl->location = options.location;
  impl->cacheable = options.cache_mode != BpuCacheMode::kNonCacheable;
  impl->alloc_flag = ToAllocFlag(options.cache_mode, options.shared_iova);
  impl->label = std::move(options.label);

#if defined(FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM)
  const char* label = impl->label.empty() ? nullptr : impl->label.c_str();
  switch (options.location) {
    case BpuMemoryLocation::kBpu:
      impl->bpu_addr = label == nullptr ? hb_bpu_mem_alloc(static_cast<std::uint64_t>(bytes), impl->alloc_flag)
                                        : hb_bpu_mem_alloc_with_label(static_cast<std::uint64_t>(bytes), impl->alloc_flag, label);
      break;
    case BpuMemoryLocation::kCpu:
      impl->bpu_addr = label == nullptr ? hb_bpu_cpumem_alloc(static_cast<std::uint64_t>(bytes), impl->alloc_flag)
                                        : hb_bpu_cpumem_alloc_with_label(static_cast<std::uint64_t>(bytes), impl->alloc_flag, label);
      break;
    case BpuMemoryLocation::kHost: {
      auto host_ptr = static_cast<std::uint8_t*>(AllocateHost(bytes));
      impl->host_data = std::shared_ptr<std::uint8_t>(host_ptr, HostDeleter{});
      break;
    }
  }

  if (options.location != BpuMemoryLocation::kHost && impl->bpu_addr == 0) {
    throw std::runtime_error("failed to allocate BPU memory");
  }
#else
  (void)options;
  auto host_ptr = static_cast<std::uint8_t*>(AllocateHost(bytes));
  impl->host_data = std::shared_ptr<std::uint8_t>(host_ptr, HostDeleter{});
  impl->location = BpuMemoryLocation::kHost;
#endif

  if (impl->location == BpuMemoryLocation::kHost && impl->host_data == nullptr) {
    throw std::runtime_error("failed to allocate host fallback memory");
  }

  return BpuBuffer{std::move(impl)};
}

BpuTensorBuffer BpuAllocator::AllocateTensor(const TensorInfo& info, BpuAllocationOptions options) const {
  BpuTensorBuffer tensor_buffer;
  tensor_buffer.info = info;
  if (options.label.empty()) {
    options.label = info.name;
  }
  tensor_buffer.buffer = Allocate(static_cast<std::size_t>(TensorStorageBytes(info)), std::move(options));
  return tensor_buffer;
}

}  // namespace foundationpose_s600
