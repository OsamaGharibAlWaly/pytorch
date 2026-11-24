// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Allocator behind torch.cuda.MemPool(backend.mem_allocator). Its segments come
// from ncclMemAlloc. They are registered with every attached ProcessGroupNCCL
// when allocated and deregistered when the pool frees them (empty_cache or the
// pool's destruction). The caching allocator calls raw_alloc/raw_delete without
// holding its own mutex, so the collective NCCL_WIN_COLL_SYMMETRIC
// (de)registration of symm attachments can run here.
//
// The caching allocator does not tell an allocator which pool it serves, so
// each MemPool needs its own NCCLMemAllocator; getMemAllocator() returns a new
// one per call.

#pragma once

#ifdef USE_C10D_NCCL

#include <memory>
#include <mutex>
#include <vector>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace c10d::nccl2 {

class NcclApi;
class ProcessGroupNCCL;

class TORCH_API NCCLMemAllocator final
    : public torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator {
 public:
  NCCLMemAllocator();

  // Registers the live segments with pg, in allocation order, and every
  // segment allocated afterwards. symm also gives them collective symmetric
  // windows, so every rank must allocate the pool's segments in the same order.
  // Attaching an already attached pg with symm=true upgrades it.
  void attach(ProcessGroupNCCL& pg, bool symm);
  // Deregisters the live segments from pg and stops registering new ones.
  // Returns false if pg was not attached.
  bool detach(ProcessGroupNCCL& pg);
  // Register or deregister the live segments with pg's current communicator,
  // keeping the attachment, as the communicator is created or torn down.
  void registerSegments(ProcessGroupNCCL& pg);
  void deregisterSegments(ProcessGroupNCCL& pg);

 private:
  struct Segment {
    void* ptr;
    size_t size;
    c10::DeviceIndex device;
  };
  // A raw pointer: a weak_intrusive_ptr would delay ~ProcessGroupNCCL until
  // the pool goes away. ProcessGroupNCCL detaches itself before destruction.
  struct Attachment {
    ProcessGroupNCCL* pg;
    bool symm;
  };

  void* allocSegment(size_t size, int device);
  void freeSegment(void* ptr, int device);
  std::vector<Attachment>::iterator findLocked(ProcessGroupNCCL& pg);
  void registerSegmentsLocked(ProcessGroupNCCL& pg, bool symm);
  void deregisterSegmentsLocked(ProcessGroupNCCL& pg);

  std::shared_ptr<NcclApi> nccl_api_;
  // Held across the NCCL calls, which take the process group's
  // memory_registration_mutex_; ProcessGroupNCCL must not call into the
  // allocator while holding that mutex.
  std::mutex mutex_;
  // In allocation order, which is the cross-rank order for collective window
  // registration.
  std::vector<Segment> segments_;
  std::vector<Attachment> attachments_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
