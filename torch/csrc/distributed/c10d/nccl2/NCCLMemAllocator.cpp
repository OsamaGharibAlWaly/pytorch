// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/NCCLMemAllocator.hpp>

#include <algorithm>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

NCCLMemAllocator::NCCLMemAllocator()
    : CUDAPluggableAllocator(
          [this](size_t size, int device, cudaStream_t /*stream*/) {
            return allocSegment(size, device);
          },
          [this](void* ptr, size_t /*size*/, int device, cudaStream_t) {
            freeSegment(ptr, device);
          }),
      nccl_api_(std::make_shared<DefaultNcclApi>()) {}

void* NCCLMemAllocator::allocSegment(size_t size, int device) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  void* ptr = nullptr;
  auto rc = nccl_api_->memAlloc(&ptr, size);
  TORCH_CHECK(
      rc == ncclSuccess,
      "ncclMemAlloc failed: ",
      nccl_api_->getErrorString(rc));
  const auto deviceIndex = static_cast<c10::DeviceIndex>(device);
  std::lock_guard<std::mutex> lock(mutex_);
  size_t registered = 0;
  try {
    for (; registered < attachments_.size(); ++registered) {
      const auto& [pg, symm] = attachments_[registered];
      pg->register_address(ptr, size, deviceIndex, symm);
    }
  } catch (...) {
    for (size_t i = 0; i < registered; ++i) {
      attachments_[i].pg->deregister_address(ptr);
    }
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->memFree(ptr), "ncclMemFree failed");
    throw;
  }
  segments_.push_back({ptr, size, deviceIndex});
  return ptr;
}

void NCCLMemAllocator::freeSegment(void* ptr, int device) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::ranges::find(segments_, ptr, &Segment::ptr);
    TORCH_INTERNAL_ASSERT(it != segments_.end(), "freeing an unknown segment");
    segments_.erase(it);
    for (const auto& attachment : attachments_) {
      attachment.pg->deregister_address(ptr);
    }
  }
  auto rc = nccl_api_->memFree(ptr);
  TORCH_CHECK(
      rc == ncclSuccess, "ncclMemFree failed: ", nccl_api_->getErrorString(rc));
}

std::vector<NCCLMemAllocator::Attachment>::iterator NCCLMemAllocator::
    findLocked(ProcessGroupNCCL& pg) {
  return std::ranges::find(attachments_, &pg, &Attachment::pg);
}

void NCCLMemAllocator::registerSegmentsLocked(ProcessGroupNCCL& pg, bool symm) {
  for (const auto& segment : segments_) {
    pg.register_address(segment.ptr, segment.size, segment.device, symm);
  }
}

void NCCLMemAllocator::deregisterSegmentsLocked(ProcessGroupNCCL& pg) {
  for (const auto& segment : segments_) {
    pg.deregister_address(segment.ptr);
  }
}

void NCCLMemAllocator::attach(ProcessGroupNCCL& pg, bool symm) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = findLocked(pg);
  if (it == attachments_.end()) {
    attachments_.push_back({&pg, symm});
    try {
      registerSegmentsLocked(pg, symm);
    } catch (...) {
      attachments_.pop_back();
      deregisterSegmentsLocked(pg);
      throw;
    }
    return;
  }
  if (it->symm == symm) {
    return;
  }
  TORCH_CHECK(
      symm,
      "MemPool is already registered with symm=True; call "
      "deregister_mem_pool before registering it with symm=False");
  registerSegmentsLocked(pg, /*symm=*/true);
  it->symm = true;
}

bool NCCLMemAllocator::detach(ProcessGroupNCCL& pg) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = findLocked(pg);
  if (it == attachments_.end()) {
    return false;
  }
  // Erase first: pg may be about to be destroyed, so it must not stay
  // reachable even if a deregistration throws.
  attachments_.erase(it);
  deregisterSegmentsLocked(pg);
  return true;
}

void NCCLMemAllocator::registerSegments(ProcessGroupNCCL& pg) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = findLocked(pg);
  if (it != attachments_.end()) {
    registerSegmentsLocked(pg, it->symm);
  }
}

void NCCLMemAllocator::deregisterSegments(ProcessGroupNCCL& pg) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (findLocked(pg) != attachments_.end()) {
    deregisterSegmentsLocked(pg);
  }
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
