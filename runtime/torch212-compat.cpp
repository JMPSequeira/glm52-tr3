// SPDX-License-Identifier: Apache-2.0
// Compatibility bridge for extensions built against the legacy no-argument API.

namespace at::cuda {
using CUDABlasHandle = void*;

CUDABlasHandle getCurrentCUDABlasHandle(bool allow_tf32);

CUDABlasHandle getCurrentCUDABlasHandle() {
    return getCurrentCUDABlasHandle(true);
}
}  // namespace at::cuda
