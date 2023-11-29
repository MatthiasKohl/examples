from cuda import cudart
import gc
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch
import torch.utils.cpp_extension as cpp_extension

import os
import subprocess
import tempfile


VALID_TYPES = ["malloc", "malloc_prefetch"]

MACROS = """
#include <cstdio>
#include <memory>
#include <stdexcept>

#define CUDA_TRY(call)                                  \
  do {                                                  \
    cudaError_t const status = call;                    \
    if (status != cudaSuccess) {                        \
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
  if (size <= 0) return nullptr;
  static constexpr ssize_t MIN_ALIGN = 256;
  // ensure that we have a valid context !
  int use_host_tables;
  auto rt_status = cudaDeviceGetAttribute(&use_host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
  if (rt_status != cudaSuccess) {
    fprintf(stderr, "Unable to get device attribute for device %d: status %d\\n", device, (int)rt_status);
    // fall back to cudaMallocManaged which should hopefully work
    void* ptr;
    cudaMallocManaged(&ptr, size);
    return ptr;
  }
  void* ptr;
  int status = posix_memalign(&ptr, MIN_ALIGN, size);
  if (__builtin_expect(status != 0, 0 /* expect false */)) {
    fprintf(stderr, "OOM for request of size %llu, aligned to %d\\n", (unsigned long long)size, (int)MIN_ALIGN);
    throw std::bad_alloc{};
  }

// TODO ! seems not ideal, but could work if simply ignoring invalid context on first CUDA call (for backward thread)
  auto adv_status = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
  if (adv_status != cudaSuccess) {
    fprintf(stderr, "cudaMemAdvise failed with %d for device %d, ignoring\\n", (int)adv_status, device);
    cudaGetLastError();
  }

  auto prefetch_status = cudaMemPrefetchAsync(ptr, size, device, stream);
  if (prefetch_status != cudaSuccess) {
    fprintf(stderr, "cudaMemPrefetchAsync failed with %d for device %d, ignoring\\n", (int)prefetch_status, device);
  }
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

class CustomTorchAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, alloc_type="malloc"):
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
        if alloc_type.lower() == "malloc":
            src = SAM_DEVICE_NOPREFETCH
        elif alloc_type.lower() == "malloc_prefetch":
            src = SAM_DEVICE_PREFETCH
        else:
            src = ""

        with os.fdopen(self.cpp_file[0], "w") as f:
            f.write(src)
        cmd.append(self.cpp_file[1])
        subprocess.check_call(" ".join(cmd), shell=True)
        super().__init__(self.so_file[1], "custom_alloc", "custom_free")

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
    if train_config.alloc_type == "rmm":
        # important: cannot use torch.cuda.current_device() to get the current
        # device since that actually initializes torch's CUDA state including
        # the default caching allocator. Instead, use cudart directly
        status, device = cudart.cudaGetDevice()
        assert status == cudart.cudaError_t.cudaSuccess,\
            "cudart.cudaGetDevice failed with " + repr(status)
        print(f"Setting up RMM allocator for device {device}")
        rmm.reinitialize(
            pool_allocator=True, managed_memory=True,
            maximum_pool_size=train_config.alloc_max_pool_size,
            devices=[device]
        )
        allocator = rmm_torch_allocator
    elif train_config.alloc_type.lower().startswith("malloc"):
        print(f"Setting up custom allocator {train_config.alloc_type}")
        allocator = CustomTorchAllocator(train_config.alloc_type)
    else:
        allocator = None

    if allocator is not None:
        torch.cuda.memory.change_current_allocator(allocator)
