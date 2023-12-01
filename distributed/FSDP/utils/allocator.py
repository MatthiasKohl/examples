from cuda import cudart
import gc
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch
import torch.utils.cpp_extension as cpp_extension

import ctypes
import os
import platform
import subprocess
import tempfile


VALID_TYPES = ["sam_device_prefetch", "sam_device_noprefetch", "sam_rmm"]

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
    // many functions may implicitly assume that CUDA allocations are always
    // aligned to at least 256 bytes: we provide the same guarantee here
    // note that `malloc` does not provide this guarantee !
    static constexpr ssize_t MIN_ALIGN = 256;
    void* ptr{nullptr};
    int status = posix_memalign(&ptr, MIN_ALIGN, bytes);
    if (COND_UNLIKELY(status != 0)) {
      fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)bytes, (int)MIN_ALIGN);
      throw std::bad_alloc{};
    }

    // this may not be ideal, but we don't know the device we're allocating on
    int device; CUDA_TRY(cudaGetDevice(&device));
    CUDA_TRY(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device));

    //if (bytes > size_t{128*1024*1024}) fprintf(stderr, "Info: allocated %llu\\n", (unsigned long long)bytes);
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

class CustomTorchAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, alloc_type, initial_pool_size=1024 * 1024,
                 max_pool_size=1024 * 1024 * 1024):
        assert alloc_type.lower() in VALID_TYPES,\
            f"Invalid allocator type {alloc_type}"
        self.cpp_file = tempfile.mkstemp(suffix=os.extsep + "cpp", text=True)
        self.so_file = tempfile.mkstemp(suffix=os.extsep + "so")
        dummy = cpp_extension.CUDAExtension("dummy", [])
        cmd = [
            "g++", "-shared", "-fPIC", "-std=c++17", "-lcudart",
            "-o", self.so_file[1]
        ]
        cmd.extend(f"-I{x}" for x in dummy.include_dirs)
        cmd.extend(f"-L{x}" for x in dummy.library_dirs)
        src = globals()[alloc_type.upper()]
        if "rmm" in alloc_type.lower():
            kwargs = {
                "initial_pool_size": f"{initial_pool_size}ULL",
                "max_pool_size": f"{max_pool_size}ULL"
            }
            # escape braces, then replace the alternate braces with actual ones
            src = src.replace("{", "{{").replace("}", "}}")
            src = src.replace("<%", "{").replace("%>", "}")
            src = src.format(**kwargs)

        with os.fdopen(self.cpp_file[0], "w") as f:
            f.write(src)
        cmd.append(self.cpp_file[1])
        subprocess.check_call(" ".join(cmd), shell=True)
        super().__init__(self.so_file[1], "custom_alloc", "custom_free")

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
            lib = ctypes.CDLL(self.so_file[1])
            dll_close(lib._handle)

    def __del__(self):
        try:
            os.remove(self.so_file[1])
        except Exception:
            pass
        try:
            os.remove(self.cpp_file[1])
        except Exception:
            pass

def setup_allocator(train_config):
    alloc_type = (train_config.alloc_type or "").lower()
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
        allocator = rmm_torch_allocator
    elif alloc_type.startswith("sam_"):
        print(f"Setting up custom allocator {alloc_type}")
        allocator = CustomTorchAllocator(
            alloc_type,
            initial_pool_size=train_config.alloc_initial_pool_size,
            max_pool_size=train_config.alloc_max_pool_size
        )
    elif alloc_type:
        raise ValueError(f"Unexpected allocator type {alloc_type}")
    else:
        allocator = None

    if allocator is not None:
        torch.cuda.memory.change_current_allocator(allocator)
        print(f"Allocator {alloc_type} set up")
    return allocator


def teardown_allocator(allocator):
    if not isinstance(allocator, CustomTorchAllocator):
        return
    allocator.teardown()
