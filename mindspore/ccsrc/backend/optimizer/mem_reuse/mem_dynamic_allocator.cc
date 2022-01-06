/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/optimizer/mem_reuse/mem_dynamic_allocator.h"
#include <string>
#include "utils/ms_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
static const char kPersistentParamMem[] = "Persistent mem";
static const char kCommonMem[] = "Common mem";
const size_t kGBToByte = 1073741824;

DynamicMemPoolBestFit::~DynamicMemPoolBestFit() {
  persistent_mem_->clear();
  common_mem_->clear();
}

DeviceMemPtr DynamicMemPoolBestFit::AllocTensorMem(size_t size, bool from_persistent_mem) {
  size_t align_size = AlignMemorySize(size);
  std::lock_guard<std::mutex> locker(mutex_);
  // Find the idle memory buf by tensor size, if not find, then add new memory block and memory buf.
  DeviceMemPtr device_addr = FindIdleMemBuf(align_size, from_persistent_mem);
  if (!device_addr) {
    device_addr = AddMemBlockAndMemBuf(align_size, from_persistent_mem);
  }
  if (!device_addr) {
    DumpDynamicMemPoolInfo();
  }
  return device_addr;
}

std::vector<DeviceMemPtr> DynamicMemPoolBestFit::AllocContinuousTensorMem(size_t total_size,
                                                                          const std::vector<size_t> &size_list) {
  std::vector<DeviceMemPtr> device_addr_list;
  // Pre-alloc the one whole piece memory.
  auto device_addr = AllocTensorMem(total_size, false);
  if (!device_addr) {
    return device_addr_list;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  // Remove the pre-alloc memory.
  const auto &mem_block = FindMemBlock(device_addr, common_mem_);
  MS_EXCEPTION_IF_NULL(mem_block);
  const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  if (mem_buf->size_ < total_size) {
    MS_LOG(EXCEPTION) << "The size of membuf is less than total_size.";
  }
  auto rest_size = mem_buf->size_ - total_size;
  (void)mem_block->block_all_mem_buf_map_.erase(iter);
  // Split the pre-alloc memory into continuous memory by the size list.
  DynamicMemBufPtr continuous_mem_buf;
  auto buf_addr = device_addr;
  for (size_t i : size_list) {
    continuous_mem_buf = std::make_shared<DynamicMemBuf>(buf_addr, kMemBufUsed, i);
    (void)mem_block->block_all_mem_buf_map_.emplace(buf_addr, continuous_mem_buf);
    device_addr_list.emplace_back(buf_addr);
    buf_addr = AddressOffset(buf_addr, i);
  }
  // Update the size of the last memory buf.
  continuous_mem_buf->size_ += rest_size;
  return device_addr_list;
}

size_t DynamicMemPoolBestFit::AlignMemorySize(size_t size) const {
  if (size == 0) {
    return DYNAMIC_MEM_ALIGN_SIZE;
  }
  return ((size + DYNAMIC_MEM_ALIGN_SIZE - 1) / DYNAMIC_MEM_ALIGN_SIZE) * DYNAMIC_MEM_ALIGN_SIZE;
}

DeviceMemPtr DynamicMemPoolBestFit::FindIdleMemBuf(size_t size, bool from_persistent_mem) {
  auto mem_mng = common_mem_;
  if (from_persistent_mem) {
    mem_mng = persistent_mem_;
  }
  MS_EXCEPTION_IF_NULL(mem_mng);
  const auto &iter = mem_mng->idle_mem_buf_map_.lower_bound(size);
  if (iter != mem_mng->idle_mem_buf_map_.end()) {
    auto mem_buf = iter->second;
    MS_EXCEPTION_IF_NULL(mem_buf);
    if (mem_buf->status_ != kMemBufIdle) {
      MS_LOG(EXCEPTION) << "Find the mem_buf is not idle, alloc_size[" << size << "] mem_buf_size[" << mem_buf->size_
                        << "] mem_buf_address[" << mem_buf->device_addr_ << "].";
    }
    mem_buf->status_ = kMemBufUsed;
    // Remove map of old idle memory buf
    (void)mem_mng->idle_mem_buf_map_.erase(iter);
    // Divide memory buf
    if (IsSplit(size, mem_buf->size_)) {
      SplitMemBuf(size, mem_buf, mem_mng);
    }
    // Memory statistics
    mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
    if (mem_mng->mps_.total_used_mem_size_ > mem_mng->mps_.used_mem_peak_size_) {
      mem_mng->mps_.used_mem_peak_size_ = mem_mng->mps_.total_used_mem_size_;
    }
    return mem_buf->device_addr_;
  }
  return nullptr;
}

size_t DynamicMemPoolBestFit::MemAllocUnitSize(bool from_persistent_mem) const {
  return from_persistent_mem ? persistent_mem_->unit_size_ : common_mem_->unit_size_;
}

void DynamicMemPoolBestFit::SetMemAllocUintSize(size_t size) {
  persistent_mem_->unit_size_ = DYNAMIC_MEM_ALLOC_UNIT_SIZE;
  common_mem_->unit_size_ = size;
  config_unit_size_ = size;
  MS_LOG(INFO) << "Set mem alloc unit size " << size;
}

void DynamicMemPoolBestFit::SetMempoolBlockSize(size_t available_device_mem_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  float mem_block_size = ms_context->get_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE);
  if (mem_block_size == kDefaultMempoolBlockSize) {
    return;
  }

  size_t config_size = FloatToSize(mem_block_size * kGBToByte);
  if (config_size > available_device_mem_size) {
    MS_LOG(WARNING) << "Memory pool block size " << config_size << " is bigger than currently available maximum memory "
                    << available_device_mem_size << ", and the actual effective value will be "
                    << available_device_mem_size;
  }
  // Reserve 1G for persistent_mem
  if (available_device_mem_size > kGBToByte) {
    available_device_mem_size -= kGBToByte;
  }
  size_t real_block_size = std::min(config_size, available_device_mem_size);
  SetMemAllocUintSize(real_block_size);
}

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem) {
  // Persistent mem is not enough, find from common
  if (from_persistent_mem && !persistent_mem_->mem_block_list_.empty()) {
    auto mem_addr = FindIdleMemBuf(size, false);
    if (mem_addr != nullptr) {
      return mem_addr;
    }
    from_persistent_mem = false;
  }
  size_t alloc_mem_size = CalMemBlockAllocSize(size, from_persistent_mem);
  if (alloc_mem_size == 0) {
    MS_LOG(DEBUG) << "Try to find in other mem";
    auto mem_addr = FindIdleMemBuf(size, !from_persistent_mem);
    if (mem_addr != nullptr) {
      return mem_addr;
    }
    return nullptr;
  }
  // Add new memory block
  DeviceMemPtr device_addr = nullptr;
  auto real_alloc_size = AllocDeviceMem(alloc_mem_size, &device_addr);
  if (real_alloc_size < size) {
    MS_LOG(WARNING) << "Memory not enough: alloc size[" << real_alloc_size << "] is smaller than required size[" << size
                    << "].";
    return nullptr;
  }
  // If unit_size is changed by other function(not context), change unit_size back
  common_mem_->unit_size_ = config_unit_size_;

  auto mem_mng = common_mem_;
  if (from_persistent_mem) {
    mem_mng = persistent_mem_;
  }
  MS_EXCEPTION_IF_NULL(mem_mng);
  auto mem_block = std::make_shared<DynamicMemBlock>(device_addr, real_alloc_size);
  MS_EXCEPTION_IF_NULL(mem_block);
  const auto &iter =
    std::upper_bound(mem_mng->mem_block_list_.begin(), mem_mng->mem_block_list_.end(), device_addr, CmpMemBlock);
  (void)mem_mng->mem_block_list_.insert(iter, mem_block);
  // Add new memory buf
  auto mem_buf = std::make_shared<DynamicMemBuf>(device_addr, kMemBufUsed, real_alloc_size);
  MS_EXCEPTION_IF_NULL(mem_buf);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(device_addr, mem_buf);
  // Split memory buf
  if (IsSplit(size, mem_buf->size_)) {
    SplitMemBuf(size, mem_buf, mem_mng);
  }
  // Memory statistics
  mem_mng->mps_.total_mem_size_ += real_alloc_size;
  mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
  if (mem_mng->mps_.total_used_mem_size_ > mem_mng->mps_.used_mem_peak_size_) {
    mem_mng->mps_.used_mem_peak_size_ = mem_mng->mps_.total_used_mem_size_;
  }
  return mem_buf->device_addr_;
}

size_t DynamicMemPoolBestFit::CalMemBlockAllocSize(size_t size, bool from_persistent_mem) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "Memory not enough: current free memory size[" << device_free_mem_size
                    << "] is smaller than required size[" << size << "].";
    return 0;
  }
  auto alloc_mem_size = MemAllocUnitSize(from_persistent_mem);
  // Growing at twice of alloc size
  constexpr size_t kDouble = 2;
  while (alloc_mem_size < size) {
    alloc_mem_size = alloc_mem_size * kDouble;
  }
  alloc_mem_size = std::min(alloc_mem_size, device_free_mem_size);
  return alloc_mem_size;
}

bool DynamicMemPoolBestFit::IsSplit(size_t tensor_size, size_t mem_buf_size) const {
  return mem_buf_size - tensor_size >= DYNAMIC_MEM_ALIGN_SIZE;
}

void DynamicMemPoolBestFit::SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf,
                                        const MemStatusManagerPtr &mem_mng) {
  MS_EXCEPTION_IF_NULL(mem_buf);
  const auto &mem_block = FindMemBlock(mem_buf->device_addr_, mem_mng);
  MS_EXCEPTION_IF_NULL(mem_block);
  // Divide new memory buf
  if (mem_buf->size_ < size) {
    MS_LOG(EXCEPTION) << "The size of membuf is less than size.";
  }
  size_t newbuf_size = mem_buf->size_ - size;
  mem_buf->size_ = size;
  DeviceMemPtr newbuf_addr = AddressOffset(mem_buf->device_addr_, size);
  auto new_mem_buf = std::make_shared<DynamicMemBuf>(newbuf_addr, kMemBufIdle, newbuf_size);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(newbuf_addr, new_mem_buf);
  // Add map of new idle memory buf
  (void)mem_mng->idle_mem_buf_map_.emplace(newbuf_size, new_mem_buf);
}

bool DynamicMemPoolBestFit::CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_block);
  return device_addr < mem_block->device_addr();
}

DynamicMemBlockPtr DynamicMemPoolBestFit::FindMemBlock(const DeviceMemPtr &device_addr,
                                                       const MemStatusManagerPtr &mem_mng) {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &&iter =
    std::upper_bound(mem_mng->mem_block_list_.begin(), mem_mng->mem_block_list_.end(), device_addr, CmpMemBlock);
  if (iter != mem_mng->mem_block_list_.begin()) {
    return *(--iter);
  }
  return nullptr;
}

void DynamicMemPoolBestFit::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
  std::lock_guard<std::mutex> locker(mutex_);
  auto fn = [this](const MemStatusManagerPtr &mem_mng, const DeviceMemPtr &device_addr) -> DynamicMemBlockPtr {
    auto mem_block = FindMemBlock(device_addr, mem_mng);
    if (mem_block != nullptr) {
      const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
      if (iter != mem_block->block_all_mem_buf_map_.end()) {
        return mem_block;
      }
    }
    return nullptr;
  };
  auto mem_block = fn(common_mem_, device_addr);
  if (mem_block == nullptr) {
    mem_block = fn(persistent_mem_, device_addr);
    if (mem_block == nullptr) {
      // Maybe destroy the memory pool first, then destroy the address, so this is normal case.
      MS_LOG(DEBUG) << "Can't find the mem_block of the device address[" << device_addr << "].";
      return;
    }
    CombineMemBuf(mem_block, device_addr, persistent_mem_);
  } else {
    CombineMemBuf(mem_block, device_addr, common_mem_);
  }
}

void DynamicMemPoolBestFit::CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceMemPtr &device_addr,
                                          const MemStatusManagerPtr &mem_mng) {
  MS_EXCEPTION_IF_NULL(mem_block);
  MS_EXCEPTION_IF_NULL(device_addr);
  const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  if (mem_buf->status_ != kMemBufUsed) {
    MS_LOG(EXCEPTION) << "Find the mem_buf is not used, mem_buf_address[" << mem_buf->device_addr_ << "].";
  }
  mem_buf->status_ = kMemBufIdle;
  if (mem_mng->mps_.total_used_mem_size_ < mem_buf->size_) {
    MS_LOG(EXCEPTION) << "The total used mem size is less than the size of membuf.";
  }
  mem_mng->mps_.total_used_mem_size_ -= mem_buf->size_;
  // Combine backward(combine the next_mem_buf to mem_buf)
  auto next_iter = iter;
  (void)next_iter++;
  if (next_iter != mem_block->block_all_mem_buf_map_.end()) {
    auto next_mem_buf = next_iter->second;
    MS_EXCEPTION_IF_NULL(next_mem_buf);
    if (next_mem_buf->status_ == kMemBufIdle) {
      mem_buf->size_ += next_mem_buf->size_;
      EraseIdleMemBuf(next_mem_buf->size_, next_mem_buf->device_addr_, mem_mng);
      (void)mem_block->block_all_mem_buf_map_.erase(next_iter);
    }
  }
  // Combine forward(combine the mem_buf to prev_mem_buf)
  bool forward_combine = false;
  DynamicMemBufPtr prev_mem_buf;
  if (iter != mem_block->block_all_mem_buf_map_.begin()) {
    auto prev_iter = iter;
    (void)prev_iter--;
    prev_mem_buf = prev_iter->second;
    MS_EXCEPTION_IF_NULL(prev_mem_buf);
    if (prev_mem_buf->status_ == kMemBufIdle) {
      EraseIdleMemBuf(prev_mem_buf->size_, prev_mem_buf->device_addr_, mem_mng);
      prev_mem_buf->size_ += mem_buf->size_;
      (void)mem_block->block_all_mem_buf_map_.erase(iter);
      forward_combine = true;
    }
  }
  // Add map of new idle memory
  if (forward_combine) {
    (void)mem_mng->idle_mem_buf_map_.emplace(prev_mem_buf->size_, prev_mem_buf);
  } else {
    (void)mem_mng->idle_mem_buf_map_.emplace(mem_buf->size_, mem_buf);
  }
}

void DynamicMemPoolBestFit::EraseIdleMemBuf(size_t size, const DeviceMemPtr &device_addr,
                                            const MemStatusManagerPtr &mem_mng) {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &&iter = mem_mng->idle_mem_buf_map_.equal_range(size);
  while (iter.first != iter.second) {
    MS_EXCEPTION_IF_NULL(iter.first->second);
    // Remove map of the idle memory buf by size and device address
    if (iter.first->second->device_addr_ == device_addr) {
      (void)mem_mng->idle_mem_buf_map_.erase(iter.first);
      return;
    }
    (void)iter.first++;
  }
  MS_LOG(ERROR) << "Can't find the size[" << size << "] and device address[" << device_addr << "] in the idle mem_buf.";
}

void DynamicMemPoolBestFit::ReleaseDeviceRes() {
  std::lock_guard<std::mutex> locker(mutex_);
  auto fn = [this](const MemStatusManagerPtr &mem_mng) {
    for (auto &iter : mem_mng->mem_block_list_) {
      auto &device_addr = iter->device_addr_base_;
      if (device_addr != nullptr) {
        if (!FreeDeviceMem(device_addr)) {
          MS_LOG(EXCEPTION) << "Free device memory[" << device_addr << "] error.";
        }
        device_addr = nullptr;
      }
    }
    mem_mng->mem_block_list_.clear();
    mem_mng->idle_mem_buf_map_.clear();
  };
  fn(common_mem_);
  fn(persistent_mem_);
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolInfo() {
  auto fn = [](const MemStatusManagerPtr &mem_mng, const std::string &mem_type) {
    if (mem_mng->mem_block_list_.empty()) {
      return;
    }
    std::ostringstream buf;
    for (size_t i = 0; i < mem_mng->mem_block_list_.size(); ++i) {
      size_t idle_size = 0;
      for (auto mb = mem_mng->mem_block_list_[i]->block_all_mem_buf_map_.begin();
           mb != mem_mng->mem_block_list_[i]->block_all_mem_buf_map_.end(); ++mb) {
        if (mb->second->status_ == kMemBufIdle) {
          idle_size += mb->second->size_;
        }
      }
      buf << ", block[" << i << "] idle size " << idle_size;
    }
    // Dump all the memory buf info
    MS_LOG(WARNING) << mem_type << "pool info: block size " << mem_mng->unit_size_ << ", block counts "
                    << mem_mng->mem_block_list_.size() << buf.str() << ". Total allocated mem "
                    << mem_mng->mps_.total_mem_size_ << ", peak used mem " << mem_mng->mps_.used_mem_peak_size_
                    << ", in used mem " << mem_mng->mps_.total_used_mem_size_ << ", total idle mem "
                    << mem_mng->mps_.total_mem_size_ - mem_mng->mps_.total_used_mem_size_;
  };
  fn(common_mem_, std::string(kCommonMem));
  fn(persistent_mem_, std::string(kPersistentParamMem));
}
}  // namespace device
}  // namespace mindspore
