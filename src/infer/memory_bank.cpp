#include "infer/memory_bank.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

namespace sam3 {

MemoryBank::MemoryBank(const MemoryBankConfig& config)
    : config_(config), next_object_id_(0) {
}

size_t MemoryBank::get_memory_size_bytes() const {
    return config_.memory_channels * config_.memory_height * config_.memory_width * sizeof(float);
}

void MemoryBank::evict_oldest_if_needed() {
    // Recent queue: 最大数を超えたら古いものを削除
    while (recent_.size() > static_cast<size_t>(config_.max_recent_frames)) {
        recent_.pop_front();
    }

    // Prompted queue: 最大数を超えたら古いものを削除
    while (prompted_.size() > static_cast<size_t>(config_.max_prompted_frames)) {
        prompted_.pop_front();
    }
}

void MemoryBank::add_frame_memory(
    int frame_id,
    const tensor::Memory<float>& spatial_memory,
    const std::vector<TrackedObject>& objects,
    bool is_prompted,
    void* stream) {

    FrameMemory frame;
    frame.frame_id = frame_id;
    frame.is_prompted = is_prompted;
    frame.objects = objects;

    // GPUメモリをコピー
    frame.spatial_memory = std::make_shared<tensor::Memory<float>>();
    size_t mem_bytes = get_memory_size_bytes();
    frame.spatial_memory->gpu(mem_bytes / sizeof(float));

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(
        frame.spatial_memory->gpu(),
        spatial_memory.gpu(),
        mem_bytes,
        cudaMemcpyDeviceToDevice,
        s
    );

    // オブジェクト情報を更新
    for (const auto& obj : objects) {
        if (active_objects_.count(obj.object_id) > 0) {
            active_objects_[obj.object_id] = obj;
            active_objects_[obj.object_id].last_seen_frame = frame_id;
        }
    }

    // 適切なキューに追加
    if (is_prompted) {
        prompted_.push_back(std::move(frame));
    } else {
        recent_.push_back(std::move(frame));
    }

    // 古いフレームを削除
    evict_oldest_if_needed();
}

void MemoryBank::gather_for_attention(
    tensor::Memory<float>& out_memories,
    tensor::Memory<float>& out_pointers,
    tensor::Memory<bool>& out_memory_mask,
    tensor::Memory<bool>& out_pointer_mask,
    int& num_memories,
    void* stream) const {

    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // 全フレーム数を計算
    num_memories = static_cast<int>(prompted_.size() + recent_.size());

    if (num_memories == 0) {
        return;
    }

    // メモリサイズを計算
    size_t single_mem_size = config_.memory_channels * config_.memory_height * config_.memory_width;
    size_t total_mem_size = num_memories * single_mem_size;
    size_t single_ptr_size = config_.max_objects_per_frame * config_.pointer_dim;
    size_t total_ptr_size = num_memories * single_ptr_size;

    // GPU/CPUメモリを確保
    out_memories.gpu(total_mem_size);
    out_pointers.gpu(total_ptr_size);
    out_pointers.cpu(total_ptr_size);
    out_memory_mask.gpu(num_memories);
    out_memory_mask.cpu(num_memories);
    out_pointer_mask.gpu(num_memories * config_.max_objects_per_frame);
    out_pointer_mask.cpu(num_memories * config_.max_objects_per_frame);

    // マスクを初期化（すべてtrue = 有効）
    bool* h_mem_mask = out_memory_mask.cpu();
    bool* h_ptr_mask = out_pointer_mask.cpu();
    for (int i = 0; i < num_memories; ++i) {
        h_mem_mask[i] = true;
    }
    for (int i = 0; i < num_memories * config_.max_objects_per_frame; ++i) {
        h_ptr_mask[i] = false;  // デフォルトは無効
    }

    // ポインタデータを初期化
    float* h_pointers = out_pointers.cpu();
    std::memset(h_pointers, 0, total_ptr_size * sizeof(float));

    int idx = 0;

    // Prompted framesを先にコピー（より重要）
    for (const auto& frame : prompted_) {
        // Spatial memoryをコピー
        float* dst = out_memories.gpu() + idx * single_mem_size;
        cudaMemcpyAsync(
            dst,
            frame.spatial_memory->gpu(),
            single_mem_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            s
        );

        // Object pointersをコピー
        int obj_idx = 0;
        for (const auto& obj : frame.objects) {
            if (obj_idx >= config_.max_objects_per_frame) break;

            float* ptr_dst = h_pointers + (idx * single_ptr_size) + (obj_idx * config_.pointer_dim);
            if (obj.pointer.size() == static_cast<size_t>(config_.pointer_dim)) {
                std::memcpy(ptr_dst, obj.pointer.data(), config_.pointer_dim * sizeof(float));
            }
            h_ptr_mask[idx * config_.max_objects_per_frame + obj_idx] = true;
            obj_idx++;
        }

        idx++;
    }

    // Recent framesをコピー
    for (const auto& frame : recent_) {
        // Spatial memoryをコピー
        float* dst = out_memories.gpu() + idx * single_mem_size;
        cudaMemcpyAsync(
            dst,
            frame.spatial_memory->gpu(),
            single_mem_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            s
        );

        // Object pointersをコピー
        int obj_idx = 0;
        for (const auto& obj : frame.objects) {
            if (obj_idx >= config_.max_objects_per_frame) break;

            float* ptr_dst = h_pointers + (idx * single_ptr_size) + (obj_idx * config_.pointer_dim);
            if (obj.pointer.size() == static_cast<size_t>(config_.pointer_dim)) {
                std::memcpy(ptr_dst, obj.pointer.data(), config_.pointer_dim * sizeof(float));
            }
            h_ptr_mask[idx * config_.max_objects_per_frame + obj_idx] = true;
            obj_idx++;
        }

        idx++;
    }

    // CPU -> GPU転送
    cudaMemcpyAsync(out_pointers.gpu(), h_pointers, total_ptr_size * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(out_memory_mask.gpu(), h_mem_mask, num_memories * sizeof(bool), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(out_pointer_mask.gpu(), h_ptr_mask, num_memories * config_.max_objects_per_frame * sizeof(bool), cudaMemcpyHostToDevice, s);
}

int MemoryBank::assign_new_object_id() {
    return next_object_id_++;
}

void MemoryBank::register_object(const TrackedObject& obj) {
    active_objects_[obj.object_id] = obj;
}

void MemoryBank::update_object_state(int object_id, bool is_active, float score, int frame_id) {
    if (active_objects_.count(object_id) > 0) {
        active_objects_[object_id].is_active = is_active;
        active_objects_[object_id].last_score = score;
        active_objects_[object_id].last_seen_frame = frame_id;
    }
}

void MemoryBank::deactivate_object(int object_id) {
    if (active_objects_.count(object_id) > 0) {
        active_objects_[object_id].is_active = false;
    }
}

std::vector<int> MemoryBank::get_active_object_ids() const {
    std::vector<int> ids;
    for (const auto& pair : active_objects_) {
        if (pair.second.is_active) {
            ids.push_back(pair.first);
        }
    }
    return ids;
}

std::vector<TrackedObject> MemoryBank::get_active_objects() const {
    std::vector<TrackedObject> objects;
    for (const auto& pair : active_objects_) {
        if (pair.second.is_active) {
            objects.push_back(pair.second);
        }
    }
    return objects;
}

bool MemoryBank::has_object(int object_id) const {
    return active_objects_.count(object_id) > 0;
}

TrackedObject* MemoryBank::get_object(int object_id) {
    auto it = active_objects_.find(object_id);
    if (it != active_objects_.end()) {
        return &it->second;
    }
    return nullptr;
}

const TrackedObject* MemoryBank::get_object(int object_id) const {
    auto it = active_objects_.find(object_id);
    if (it != active_objects_.end()) {
        return &it->second;
    }
    return nullptr;
}

void MemoryBank::reset() {
    recent_.clear();
    prompted_.clear();
    active_objects_.clear();
    next_object_id_ = 0;
}

void MemoryBank::clear_recent() {
    recent_.clear();
}

} // namespace sam3
