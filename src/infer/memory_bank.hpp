#ifndef MEMORY_BANK_HPP__
#define MEMORY_BANK_HPP__

#include <deque>
#include <vector>
#include <unordered_map>
#include <array>
#include <memory>
#include <string>
#include "common/memory.hpp"

namespace sam3 {

// 単一オブジェクトの追跡情報
struct TrackedObject {
    int object_id;
    std::string class_name;
    float last_score;
    std::vector<float> pointer;  // オブジェクトポインタ [256]
    std::array<float, 4> last_bbox;  // x1, y1, x2, y2
    bool is_active;
    int last_seen_frame;

    TrackedObject() : object_id(-1), last_score(0.0f), is_active(false), last_seen_frame(-1) {
        pointer.resize(256, 0.0f);
        last_bbox = {0, 0, 0, 0};
    }
};

// フレームメモリ
struct FrameMemory {
    int frame_id;
    bool is_prompted;
    std::shared_ptr<tensor::Memory<float>> spatial_memory;  // [256, 72, 72]
    std::vector<TrackedObject> objects;

    FrameMemory() : frame_id(-1), is_prompted(false) {}
};

// Memory Bank設定
struct MemoryBankConfig {
    int max_recent_frames = 6;      // 最新フレームキャッシュ数
    int max_prompted_frames = 2;    // プロンプト付きフレーム数
    int max_objects_per_frame = 10; // フレームあたり最大オブジェクト数
    int memory_channels = 256;
    int memory_height = 72;
    int memory_width = 72;
    int pointer_dim = 256;          // オブジェクトポインタ次元
};

class MemoryBank {
public:
    explicit MemoryBank(const MemoryBankConfig& config = MemoryBankConfig());
    ~MemoryBank() = default;

    // メモリ追加
    void add_frame_memory(
        int frame_id,
        const tensor::Memory<float>& spatial_memory,
        const std::vector<TrackedObject>& objects,
        bool is_prompted,
        void* stream = nullptr
    );

    // Memory Attention用にメモリを取得（GPU上にGather）
    void gather_for_attention(
        tensor::Memory<float>& out_memories,      // [N, 256, 72, 72]
        tensor::Memory<float>& out_pointers,      // [N, max_obj, 256]
        tensor::Memory<bool>& out_memory_mask,    // [N]
        tensor::Memory<bool>& out_pointer_mask,   // [N, max_obj]
        int& num_memories,
        void* stream = nullptr
    ) const;

    // オブジェクトID管理
    int assign_new_object_id();
    void register_object(const TrackedObject& obj);
    void update_object_state(int object_id, bool is_active, float score, int frame_id);
    void deactivate_object(int object_id);

    // オブジェクト情報取得
    std::vector<int> get_active_object_ids() const;
    std::vector<TrackedObject> get_active_objects() const;
    bool has_object(int object_id) const;
    TrackedObject* get_object(int object_id);
    const TrackedObject* get_object(int object_id) const;

    // リセット
    void reset();
    void clear_recent();

    // 状態確認
    int get_frame_count() const { return static_cast<int>(recent_.size() + prompted_.size()); }
    bool empty() const { return recent_.empty() && prompted_.empty(); }
    int get_recent_count() const { return static_cast<int>(recent_.size()); }
    int get_prompted_count() const { return static_cast<int>(prompted_.size()); }

    // 設定取得
    const MemoryBankConfig& get_config() const { return config_; }

private:
    // 内部ヘルパー
    size_t get_memory_size_bytes() const;
    void evict_oldest_if_needed();

    MemoryBankConfig config_;
    std::deque<FrameMemory> recent_;     // 最新フレーム（FIFO）
    std::deque<FrameMemory> prompted_;   // プロンプト付きフレーム（FIFO）
    std::unordered_map<int, TrackedObject> active_objects_;  // アクティブオブジェクト

    int next_object_id_ = 0;
};

} // namespace sam3

#endif // MEMORY_BANK_HPP__
