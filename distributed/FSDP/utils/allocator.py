from cuda import cudart
import gc
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch
import torch.utils.cpp_extension as cpp_extension

from contextlib import contextmanager
import ctypes
from enum import Enum
import os
import platform
import subprocess
import tempfile


VALID_TYPES = ["sam_device_prefetch", "sam_device_noprefetch", "sam_rmm", "sam_rmm_managed"]

class DeviceType(Enum):
    DEFAULT = (0, "cudaMemLocationTypeInvalid")
    DEVICE = (1, "cudaMemLocationTypeDevice")
    HOST = (2, "cudaMemLocationTypeHost")


MACROS = """
#include <cstdio>
#include <memory>
#include <stdexcept>

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define COND_LIKELY(expr)   (__builtin_expect(static_cast<bool>(expr), 1))
#define COND_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define COND_LIKELY(expr)   static_cast<bool>(expr)
#define COND_UNLIKELY(expr) static_cast<bool>(expr)
#endif

#define CUDA_TRY(call)                                  \
  do {                                                  \
    cudaError_t const status = call;                    \
    if (COND_UNLIKELY(status != cudaSuccess)) {         \
      cudaGetLastError();                               \
      auto error_name = cudaGetErrorName(status);       \
      auto error_str = cudaGetErrorString(status);      \
      auto msg = "CUDA error for '%s', reason=%s:%s";   \
      auto size = std::snprintf(                        \
        nullptr, 0, msg, #call, error_name, error_str); \
      auto buf = std::make_unique<char[]>(              \
        size_t(size + 1));                              \
      std::snprintf(                                    \
        buf.get(), size + 1, msg, #call, error_name,    \
        error_str);                                     \
      throw std::runtime_error(buf.get());              \
    }                                                   \
  } while (0)
"""

SAM_DEVICE_PREFETCH = MACROS + """
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <sys/types.h>

extern "C" {
void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  if (COND_UNLIKELY(size <= 0)) return nullptr;
  // many functions may implicitly assume that CUDA allocations are always
  // aligned to at least 256 bytes: we provide the same guarantee here
  // note that `malloc` does not provide this guarantee !
  static constexpr ssize_t MIN_ALIGN = 256;
  void* ptr;
  int status = posix_memalign(&ptr, MIN_ALIGN, size);
  if (COND_UNLIKELY(status != 0)) {
    fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)size, (int)MIN_ALIGN);
    throw std::bad_alloc{};
  }

  // backward thread seems to sometimes not have device context initialized
  // before calling into this: ignore that error and just try again
  auto adv_status = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
  if (COND_UNLIKELY(adv_status == cudaErrorDeviceUninitialized)) {
    // ignore last error, set device (make primary context current to this thread)
    // and then try again
    cudaGetLastError();
    CUDA_TRY(cudaSetDevice(device));
    CUDA_TRY(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
  } else {
    CUDA_TRY(adv_status);
  }

  CUDA_TRY(cudaMemPrefetchAsync(ptr, size, device, stream));
  return ptr;
}
void custom_free(void* ptr, ssize_t /*size*/, int /*device*/, cudaStream_t /*stream*/) {
  std::free(ptr);
}
}
"""

SAM_DEVICE_NOPREFETCH = os.linesep.join(
    line for line in SAM_DEVICE_PREFETCH.split(os.linesep)
    if "cudaMemPrefetchAsync" not in line
)

SAM_RMM = MACROS + """
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

static enum cudaMemLocationType prefetch = cudaMemLocationTypeInvalid;
static enum cudaMemLocationType location = cudaMemLocationTypeInvalid;
static enum cudaMemLocationType accessed_by = cudaMemLocationTypeInvalid;

class sam_device_memory_resource final : public rmm::mr::device_memory_resource {
  public:
  sam_device_memory_resource()                                  = default;
  ~sam_device_memory_resource() override                        = default;
  sam_device_memory_resource(sam_device_memory_resource const&) = default;  ///< @default_copy_constructor
  sam_device_memory_resource(sam_device_memory_resource&&)      = default;  ///< @default_move_constructor
  sam_device_memory_resource& operator=(sam_device_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_device_memory_resource}
  sam_device_memory_resource& operator=(sam_device_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_device_memory_resource}

  [[nodiscard]] bool supports_streams() const noexcept override { return false; }
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  private:
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (COND_UNLIKELY(bytes <= 0)) return nullptr;
    // the pool allocator should get (huge) page-aligned allocations
    static constexpr ssize_t MIN_ALIGN = 1 << 21;
    void* ptr{nullptr};
    int status = posix_memalign(&ptr, MIN_ALIGN, bytes);
    if (COND_UNLIKELY(status != 0)) {
      fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)bytes, (int)MIN_ALIGN);
      throw std::bad_alloc{};
    }

    // this may not be ideal, but we don't know the device we're allocating on
    int device; CUDA_TRY(cudaGetDevice(&device));
    if (<%pool_location%> != cudaMemLocationTypeInvalid) {
        int dst = <%pool_location%> == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, dst));
    }
    if (<%pool_accessed_by%> != cudaMemLocationTypeInvalid) {
        int dst = <%pool_accessed_by%> == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, dst));
    }
    if (<%pool_prefetch%> != cudaMemLocationTypeInvalid) {
        int dst = <%pool_prefetch%> == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemPrefetchAsync(ptr, bytes, dst));
    }

    if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: freeing %llu\\n", (unsigned long long)bytes);
    free(ptr);
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<sam_device_memory_resource const*>(&other) != nullptr;
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

extern "C" {

sam_device_memory_resource* sam_resource{nullptr};
typedef rmm::mr::pool_memory_resource<sam_device_memory_resource> pool_resource;
pool_resource** pool_resources{nullptr};
int device_count{0};

static void __attribute__((constructor)) init() {
  sam_resource = new sam_device_memory_resource{};
  auto initial_pool_size = size_t{<%initial_pool_size%>};
  auto max_pool_size = size_t{<%max_pool_size%>};
  CUDA_TRY(cudaGetDeviceCount(&device_count));
  pool_resources = new pool_resource*[device_count];
  printf("Allocator: %d devices, init pool size %llu, max pool size %llu\\n",
    device_count, (unsigned long long)initial_pool_size,
    (unsigned long long)max_pool_size);
  for (int device = 0; device < device_count; ++device) {
    CUDA_TRY(cudaSetDevice(device));
    pool_resources[device] = new pool_resource(
      sam_resource, initial_pool_size, max_pool_size
    );
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device}, pool_resources[device]);
  }
}

static void __attribute__((destructor)) finalize() {
  for (int device = 0; device < device_count; ++device) {
    delete pool_resources[device];
  }
  delete[] pool_resources;
  delete sam_resource;
}

void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  auto* ptr = mr->allocate(size, rmm::cuda_stream_view{stream});
  if (COND_UNLIKELY(ptr == nullptr || size == 0)) return ptr;
  if (location != cudaMemLocationTypeInvalid || accessed_by != cudaMemLocationTypeInvalid || prefetch != cudaMemLocationTypeInvalid) {
    // make sure that device context is set
    CUDA_TRY(cudaSetDevice(device));
    // advise/prefetch should always be aligned to 2MB here
    static constexpr ssize_t MIN_ALIGN = 1 << 21;
    auto start_ptr = (reinterpret_cast<size_t>(ptr) / MIN_ALIGN) * MIN_ALIGN;
    auto end_ptr = reinterpret_cast<size_t>(ptr) + size;
    end_ptr = ((end_ptr + MIN_ALIGN - 1) / MIN_ALIGN) * MIN_ALIGN;
    if (COND_UNLIKELY(location != cudaMemLocationTypeInvalid)) {
        //printf("alloc: location %d for %llu bytes, ptr %p, page-aligned %p/%llu\\n", int(location), (unsigned long long)size, ptr, reinterpret_cast<void*>(start_ptr), (unsigned long long)(end_ptr - start_ptr));
        int dst = location == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemAdvise(reinterpret_cast<void*>(start_ptr), end_ptr - start_ptr, cudaMemAdviseSetPreferredLocation, dst));
    }
    if (COND_UNLIKELY(accessed_by != cudaMemLocationTypeInvalid)) {
        //printf("alloc: accessed_by %d for %llu bytes, ptr %p, page-aligned %p/%llu\\n", int(accessed_by), (unsigned long long)size, ptr, reinterpret_cast<void*>(start_ptr), (unsigned long long)(end_ptr - start_ptr));
        int dst = accessed_by == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemAdvise(reinterpret_cast<void*>(start_ptr), end_ptr - start_ptr, cudaMemAdviseSetAccessedBy, dst));
    }
    if (COND_UNLIKELY(prefetch != cudaMemLocationTypeInvalid)) {
        //printf("alloc: prefetch %d for %llu bytes, ptr %p, page-aligned %p/%llu\\n", int(prefetch), (unsigned long long)size, ptr, reinterpret_cast<void*>(start_ptr), (unsigned long long)(end_ptr - start_ptr));
        int dst = prefetch == cudaMemLocationTypeDevice ? device : cudaCpuDeviceId;
        CUDA_TRY(cudaMemPrefetchAsync(reinterpret_cast<void*>(start_ptr), end_ptr - start_ptr, dst));
    }
  }
  return ptr;
}

void custom_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  mr->deallocate(ptr, size, rmm::cuda_stream_view{stream});
}

int get_location() { return static_cast<int>(location); }
int get_accessed_by() { return static_cast<int>(accessed_by); }
int get_prefetch() { return static_cast<int>(prefetch); }
void set_location(int value) { location = static_cast<enum cudaMemLocationType>(value); }
void set_accessed_by(int value) { accessed_by = static_cast<enum cudaMemLocationType>(value); }
void set_prefetch(int value) { prefetch = static_cast<enum cudaMemLocationType>(value); }

}
"""

SAM_RMM_DEFAULT = MACROS + """
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

class sam_device_memory_resource final : public rmm::mr::device_memory_resource {
  public:
  sam_device_memory_resource()                                  = default;
  ~sam_device_memory_resource() override                        = default;
  sam_device_memory_resource(sam_device_memory_resource const&) = default;  ///< @default_copy_constructor
  sam_device_memory_resource(sam_device_memory_resource&&)      = default;  ///< @default_move_constructor
  sam_device_memory_resource& operator=(sam_device_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_device_memory_resource}
  sam_device_memory_resource& operator=(sam_device_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_device_memory_resource}

  [[nodiscard]] bool supports_streams() const noexcept override { return false; }
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  private:
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (COND_UNLIKELY(bytes <= 0)) return nullptr;
    // the pool allocator should get (huge) page-aligned allocations
    static constexpr ssize_t MIN_ALIGN = 1 << 21;
    void* ptr{nullptr};
    int status = posix_memalign(&ptr, MIN_ALIGN, bytes);
    if (COND_UNLIKELY(status != 0)) {
      fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)bytes, (int)MIN_ALIGN);
      throw std::bad_alloc{};
    }

    if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: freeing %llu\\n", (unsigned long long)bytes);
    free(ptr);
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<sam_device_memory_resource const*>(&other) != nullptr;
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

extern "C" {

sam_device_memory_resource* sam_resource{nullptr};
typedef rmm::mr::pool_memory_resource<sam_device_memory_resource> pool_resource;
pool_resource** pool_resources{nullptr};
int device_count{0};

static void __attribute__((constructor)) init() {
  sam_resource = new sam_device_memory_resource{};
  auto initial_pool_size = size_t{<%initial_pool_size%>};
  auto max_pool_size = size_t{<%max_pool_size%>};
  CUDA_TRY(cudaGetDeviceCount(&device_count));
  pool_resources = new pool_resource*[device_count];
  printf("Allocator: %d devices, init pool size %llu, max pool size %llu\\n",
    device_count, (unsigned long long)initial_pool_size,
    (unsigned long long)max_pool_size);
  for (int device = 0; device < device_count; ++device) {
    CUDA_TRY(cudaSetDevice(device));
    pool_resources[device] = new pool_resource(
      sam_resource, initial_pool_size, max_pool_size
    );
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device}, pool_resources[device]);
  }
}

static void __attribute__((destructor)) finalize() {
  for (int device = 0; device < device_count; ++device) {
    delete pool_resources[device];
  }
  delete[] pool_resources;
  delete sam_resource;
}

void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  return mr->allocate(size, rmm::cuda_stream_view{stream});
}
void custom_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  mr->deallocate(ptr, size, rmm::cuda_stream_view{stream});
}
}
"""

SAM_RMM_CPU = MACROS + """
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

class sam_device_memory_resource final : public rmm::mr::device_memory_resource {
  public:
  sam_device_memory_resource()                                  = default;
  ~sam_device_memory_resource() override                        = default;
  sam_device_memory_resource(sam_device_memory_resource const&) = default;  ///< @default_copy_constructor
  sam_device_memory_resource(sam_device_memory_resource&&)      = default;  ///< @default_move_constructor
  sam_device_memory_resource& operator=(sam_device_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_device_memory_resource}
  sam_device_memory_resource& operator=(sam_device_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_device_memory_resource}

  [[nodiscard]] bool supports_streams() const noexcept override { return false; }
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  private:
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (COND_UNLIKELY(bytes <= 0)) return nullptr;
    // the pool allocator should get (huge) page-aligned allocations
    static constexpr ssize_t MIN_ALIGN = 1 << 21;
    void* ptr{nullptr};
    int status = posix_memalign(&ptr, MIN_ALIGN, bytes);
    if (COND_UNLIKELY(status != 0)) {
      fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)bytes, (int)MIN_ALIGN);
      throw std::bad_alloc{};
    }

    int device; CUDA_TRY(cudaGetDevice(&device));
    CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device));

    if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: freeing %llu\\n", (unsigned long long)bytes);
    free(ptr);
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<sam_device_memory_resource const*>(&other) != nullptr;
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

extern "C" {

sam_device_memory_resource* sam_resource{nullptr};
typedef rmm::mr::pool_memory_resource<sam_device_memory_resource> pool_resource;
pool_resource** pool_resources{nullptr};
int device_count{0};

static void __attribute__((constructor)) init() {
  sam_resource = new sam_device_memory_resource{};
  auto initial_pool_size = size_t{<%initial_pool_size%>};
  auto max_pool_size = size_t{<%max_pool_size%>};
  CUDA_TRY(cudaGetDeviceCount(&device_count));
  pool_resources = new pool_resource*[device_count];
  printf("Allocator: %d devices, init pool size %llu, max pool size %llu\\n",
    device_count, (unsigned long long)initial_pool_size,
    (unsigned long long)max_pool_size);
  for (int device = 0; device < device_count; ++device) {
    CUDA_TRY(cudaSetDevice(device));
    pool_resources[device] = new pool_resource(
      sam_resource, initial_pool_size, max_pool_size
    );
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device}, pool_resources[device]);
  }
}

static void __attribute__((destructor)) finalize() {
  for (int device = 0; device < device_count; ++device) {
    delete pool_resources[device];
  }
  delete[] pool_resources;
  delete sam_resource;
}

void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  return mr->allocate(size, rmm::cuda_stream_view{stream});
}
void custom_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  mr->deallocate(ptr, size, rmm::cuda_stream_view{stream});
}
}
"""

SAM_RMM_MANAGED = MACROS + """
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>

class sam_device_memory_resource final : public rmm::mr::device_memory_resource {
  public:
  sam_device_memory_resource()                                  = default;
  ~sam_device_memory_resource() override                        = default;
  sam_device_memory_resource(sam_device_memory_resource const&) = default;  ///< @default_copy_constructor
  sam_device_memory_resource(sam_device_memory_resource&&)      = default;  ///< @default_move_constructor
  sam_device_memory_resource& operator=(sam_device_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_device_memory_resource}
  sam_device_memory_resource& operator=(sam_device_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_device_memory_resource}

  [[nodiscard]] bool supports_streams() const noexcept override { return false; }
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  private:
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (COND_UNLIKELY(bytes <= 0)) return nullptr;
    void* ptr{nullptr};
    CUDA_TRY(cudaMallocManaged(&ptr, bytes));
    int device; CUDA_TRY(cudaGetDevice(&device));
    CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device));

    if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: freeing %llu\\n", (unsigned long long)bytes);
    CUDA_TRY(cudaFree(ptr));
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<sam_device_memory_resource const*>(&other) != nullptr;
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

extern "C" {

sam_device_memory_resource* sam_resource{nullptr};
typedef rmm::mr::pool_memory_resource<sam_device_memory_resource> pool_resource;
pool_resource** pool_resources{nullptr};
int device_count{0};

static void __attribute__((constructor)) init() {
  sam_resource = new sam_device_memory_resource{};
  auto initial_pool_size = size_t{<%initial_pool_size%>};
  auto max_pool_size = size_t{<%max_pool_size%>};
  CUDA_TRY(cudaGetDeviceCount(&device_count));
  pool_resources = new pool_resource*[device_count];
  printf("Allocator: %d devices, init pool size %llu, max pool size %llu\\n",
    device_count, (unsigned long long)initial_pool_size,
    (unsigned long long)max_pool_size);
  for (int device = 0; device < device_count; ++device) {
    CUDA_TRY(cudaSetDevice(device));
    pool_resources[device] = new pool_resource(
      sam_resource, initial_pool_size, max_pool_size
    );
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device}, pool_resources[device]);
  }
}

static void __attribute__((destructor)) finalize() {
  for (int device = 0; device < device_count; ++device) {
    delete pool_resources[device];
  }
  delete[] pool_resources;
  delete sam_resource;
}

void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  return mr->allocate(size, rmm::cuda_stream_view{stream});
}
void custom_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  mr->deallocate(ptr, size, rmm::cuda_stream_view{stream});
}
}
"""

SAM_RMM_MANAGED_DEFAULT = MACROS + """
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>

class sam_device_memory_resource final : public rmm::mr::device_memory_resource {
  public:
  sam_device_memory_resource()                                  = default;
  ~sam_device_memory_resource() override                        = default;
  sam_device_memory_resource(sam_device_memory_resource const&) = default;  ///< @default_copy_constructor
  sam_device_memory_resource(sam_device_memory_resource&&)      = default;  ///< @default_move_constructor
  sam_device_memory_resource& operator=(sam_device_memory_resource const&) =
    default;  ///< @default_copy_assignment{sam_device_memory_resource}
  sam_device_memory_resource& operator=(sam_device_memory_resource&&) =
    default;  ///< @default_move_assignment{sam_device_memory_resource}

  [[nodiscard]] bool supports_streams() const noexcept override { return false; }
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  private:
  void* do_allocate(std::size_t bytes, [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (COND_UNLIKELY(bytes <= 0)) return nullptr;
    void* ptr{nullptr};
    CUDA_TRY(cudaMallocManaged(&ptr, bytes));

    if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: freeing %llu\\n", (unsigned long long)bytes);
    CUDA_TRY(cudaFree(ptr));
  }

  [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<sam_device_memory_resource const*>(&other) != nullptr;
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    return std::make_pair(0, 0);
  }
};

extern "C" {

sam_device_memory_resource* sam_resource{nullptr};
typedef rmm::mr::pool_memory_resource<sam_device_memory_resource> pool_resource;
pool_resource** pool_resources{nullptr};
int device_count{0};

static void __attribute__((constructor)) init() {
  sam_resource = new sam_device_memory_resource{};
  auto initial_pool_size = size_t{<%initial_pool_size%>};
  auto max_pool_size = size_t{<%max_pool_size%>};
  CUDA_TRY(cudaGetDeviceCount(&device_count));
  pool_resources = new pool_resource*[device_count];
  printf("Allocator: %d devices, init pool size %llu, max pool size %llu\\n",
    device_count, (unsigned long long)initial_pool_size,
    (unsigned long long)max_pool_size);
  for (int device = 0; device < device_count; ++device) {
    CUDA_TRY(cudaSetDevice(device));
    pool_resources[device] = new pool_resource(
      sam_resource, initial_pool_size, max_pool_size
    );
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device}, pool_resources[device]);
  }
}

static void __attribute__((destructor)) finalize() {
  for (int device = 0; device < device_count; ++device) {
    delete pool_resources[device];
  }
  delete[] pool_resources;
  delete sam_resource;
}

void* custom_alloc(ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  return mr->allocate(size, rmm::cuda_stream_view{stream});
}
void custom_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device});
  mr->deallocate(ptr, size, rmm::cuda_stream_view{stream});
}
}
"""

class CustomTorchAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, alloc_type, initial_pool_size=1024 * 1024,
                 max_pool_size=1024 * 1024 * 1024,
                 pool_location=DeviceType.DEFAULT,
                 pool_accessed_by=DeviceType.DEFAULT,
                 pool_prefetch=DeviceType.DEFAULT):
        assert alloc_type.lower() in VALID_TYPES,\
            f"Invalid allocator type {alloc_type}"
        self.cpp_file = tempfile.mkstemp(suffix=os.extsep + "cpp", text=True)
        self.so_file = tempfile.mkstemp(suffix=os.extsep + "so")
        self.alloc = self.so_file[1]
        dummy = cpp_extension.CUDAExtension("dummy", [])
        cmd = [
            "g++", "-shared", "-fPIC", "-std=c++17", "-lcudart", "-DNDEBUG",
            "-o", self.alloc
        ]
        cmd.extend(f"-I{x}" for x in dummy.include_dirs)
        cmd.extend(f"-L{x}" for x in dummy.library_dirs)
        src = globals()[alloc_type.upper()]
        if "rmm" in alloc_type.lower():
            kwargs = {
                "initial_pool_size": f"{initial_pool_size}ULL",
                "max_pool_size": f"{max_pool_size}ULL",
                "pool_location": pool_location.value[1],
                "pool_accessed_by": pool_accessed_by.value[1],
                "pool_prefetch": pool_prefetch.value[1],
            }
            # escape braces, then replace the alternate braces with actual ones
            src = src.replace("{", "{{").replace("}", "}}")
            src = src.replace("<%", "{").replace("%>", "}")
            src = src.format(**kwargs)

        with os.fdopen(self.cpp_file[0], "w") as f:
            f.write(src)
        cmd.append(self.cpp_file[1])
        subprocess.check_call(" ".join(cmd), shell=True)
        super().__init__(self.alloc, "custom_alloc", "custom_free")

    def teardown(self):
        # attempt to actually close / unload the library
        # this is to ensure that the CUDA driver is still running
        # in the process when garbage collecting an instance of this class
        if platform.system() == "Linux":
            try:
                stdlib = ctypes.CDLL("")
            except OSError:
                # Alpine Linux.
                stdlib = ctypes.CDLL("libc.so")
            dll_close = stdlib.dlclose
            dll_close.argtypes = [ctypes.c_void_p]
            lib = ctypes.CDLL(self.alloc)
            dll_close(lib._handle)

    def __del__(self):
        try:
            os.remove(self.alloc)
        except Exception:
            pass
        try:
            os.remove(self.alloc)
        except Exception:
            pass

    @contextmanager
    def use(self,
            location=DeviceType.DEFAULT,
            accessed_by=DeviceType.DEFAULT,
            prefetch=DeviceType.DEFAULT):
        # get current values
        cdll = ctypes.CDLL(self.alloc)
        current_values = []
        for kind, val in [
                    ("location", location),
                    ("accessed_by", accessed_by),
                    ("prefetch", prefetch)
                ]:
            get_func = getattr(cdll, "get_" + kind)
            get_func.restype = ctypes.c_int
            get_func.argtypes = []
            set_func = getattr(cdll, "set_" + kind)
            set_func.restype = None
            set_func.argtypes = [ctypes.c_int]
            # record current value
            current_val = get_func()
            # apply new value
            set_func(val.value[0])
            current_values.append((set_func, current_val))

        try:
            yield self.alloc
        finally:
            # reset old values
            for set_func, current_val in current_values:
                set_func(current_val)


def setup_allocator(train_config):
    alloc_type = (train_config.alloc_type or "").lower()
    class DummyAllocator(object):
        def __init__(self, alloc=None):
            self.alloc = alloc

        @contextmanager
        def use(self,
                location=DeviceType.DEFAULT,
                accessed_by=DeviceType.DEFAULT,
                prefetch=DeviceType.DEFAULT):
            yield self.alloc

    if alloc_type == "rmm":
        # important: cannot use torch.cuda.current_device() to get the current
        # device since that actually initializes torch's CUDA state including
        # the default caching allocator. Instead, use cudart directly
        status, device = cudart.cudaGetDevice()
        assert status == cudart.cudaError_t.cudaSuccess,\
            "cudart.cudaGetDevice failed with " + repr(status)
        print(f"Setting up RMM allocator for device {device}")
        rmm.reinitialize(
            pool_allocator=True, managed_memory=True,
            initial_pool_size=train_config.alloc_initial_pool_size,
            maximum_pool_size=train_config.alloc_max_pool_size,
            devices=[device]
        )
        allocator = DummyAllocator(rmm_torch_allocator)
    elif alloc_type.startswith("sam_"):
        print(f"Setting up custom allocator {alloc_type}")
        allocator = CustomTorchAllocator(
            alloc_type,
            initial_pool_size=train_config.alloc_initial_pool_size,
            max_pool_size=train_config.alloc_max_pool_size,
            pool_location=getattr(DeviceType, train_config.pool_location.upper()),
            pool_accessed_by=getattr(DeviceType, train_config.pool_accessed_by.upper()),
            pool_prefetch=getattr(DeviceType, train_config.pool_prefetch.upper())
        )
    elif alloc_type:
        raise ValueError(f"Unexpected allocator type {alloc_type}")
    else:
        allocator = DummyAllocator()

    if allocator.alloc is not None:
        torch.cuda.memory.change_current_allocator(allocator)
        print(f"Allocator {alloc_type} set up")
    return allocator


def teardown_allocator(allocator):
    if not isinstance(allocator, CustomTorchAllocator):
        return
    allocator.teardown()
