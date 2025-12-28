#ifndef SAM3_TRACKER_HPP__
#define SAM3_TRACKER_HPP__

#include "infer/sam3infer.hpp"
#include "infer/memory_bank.hpp"
#include "common/tensorrt.hpp"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

namespace sam3 {

// 追跡結果
struct TrackingResult {
    int frame_id;
    int object_id;
    std::string class_name;
    float confidence;
    std::array<float, 4> bbox;  // x1, y1, x2, y2
    cv::Mat mask;               // セグメンテーションマスク
    bool is_new;                // 新規検出か追跡継続か

    TrackingResult()
        : frame_id(-1), object_id(-1), confidence(0.0f), is_new(false) {
        bbox = {0, 0, 0, 0};
    }
};

// SAM3 Tracker設定
struct Sam3TrackerConfig {
    // エンジンパス
    std::string vision_encoder_path;
    std::string text_encoder_path;
    std::string geometry_encoder_path;
    std::string decoder_path;
    std::string memory_encoder_path;
    std::string memory_attention_path;

    // 追跡設定
    int gpu_id = 0;
    float detection_threshold = 0.5f;
    float tracking_threshold = 0.3f;   // 追跡継続の閾値
    float match_iou_threshold = 0.5f;  // 検出-追跡マッチングIoU閾値
    int max_lost_frames = 30;          // オブジェクト消失許容フレーム数

    // Memory Bank設定
    MemoryBankConfig memory_config;

    Sam3TrackerConfig() = default;
};

class Sam3Tracker {
public:
    static std::shared_ptr<Sam3Tracker> create_instance(const Sam3TrackerConfig& config);

    virtual ~Sam3Tracker() = default;

    // 初期化（最初のフレームでプロンプト設定）
    bool initialize(
        const cv::Mat& first_frame,
        const std::vector<Sam3PromptUnit>& prompts,
        float confidence_threshold = 0.5f
    );

    // ビデオフレーム追跡
    std::vector<TrackingResult> track_frame(const cv::Mat& frame);

    // インタラクティブ修正
    bool add_correction(
        int frame_id,
        int object_id,
        const Sam3PromptUnit& prompt
    );

    // プロンプト追加（新しいオブジェクトタイプを追跡）
    bool add_prompt(const Sam3PromptUnit& prompt);

    // オブジェクト管理
    bool remove_object(int object_id);
    std::vector<int> get_tracked_object_ids() const;
    std::vector<TrackedObject> get_tracked_objects() const;

    // リセット
    void reset();

    // 状態取得
    int get_current_frame_id() const { return current_frame_id_; }
    bool is_initialized() const { return is_initialized_; }
    bool has_memory_engines() const { return has_memory_engines_; }

    // 設定取得/設定
    const Sam3TrackerConfig& get_config() const { return config_; }
    void set_detection_threshold(float threshold) { config_.detection_threshold = threshold; }
    void set_tracking_threshold(float threshold) { config_.tracking_threshold = threshold; }

protected:
    Sam3Tracker(const Sam3TrackerConfig& config);

    bool load_engines();
    void allocate_buffers();

    // 推論パイプライン
    void preprocess_frame(const cv::Mat& frame, void* stream);
    void encode_vision(void* stream);
    void apply_memory_attention(void* stream);
    std::vector<TrackingResult> run_detection(void* stream);
    void run_tracking_propagation(void* stream);
    void encode_memory(bool is_prompted, void* stream);

    // マッチングと更新
    void match_and_update(
        std::vector<TrackingResult>& detection_results,
        std::vector<TrackingResult>& tracking_results
    );

    // IoU計算
    float compute_iou(const std::array<float, 4>& box1, const std::array<float, 4>& box2) const;

    // オブジェクトポインタ抽出
    std::vector<TrackedObject> extract_object_pointers(
        const std::vector<TrackingResult>& results
    ) const;

    // TRT dimension設定
    void set_binding_dim(std::shared_ptr<TensorRT::Engine>& engine, int idx, const std::vector<int>& dims);

protected:
    // 設定
    Sam3TrackerConfig config_;

    // エンジン
    std::shared_ptr<Sam3Infer> detector_;  // 既存のDetector
    std::shared_ptr<TensorRT::Engine> memory_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> memory_attention_trt_;

    // メモリ管理
    std::unique_ptr<MemoryBank> memory_bank_;

    // 入力バッファ（Vision Encoderの出力を保持）
    tensor::Memory<float> vision_features_;        // [1, 256, 72, 72] fpn_feat_2
    tensor::Memory<float> vision_features_flat_;   // [1, 5184, 256] for memory attention

    // Memory Attentionバッファ
    tensor::Memory<float> conditioned_features_;   // [1, 5184, 256]
    tensor::Memory<float> gathered_memories_;      // [N, 256, 72, 72]
    tensor::Memory<float> gathered_pointers_;      // [N, max_obj, 256]
    tensor::Memory<bool> memory_mask_;             // [N]
    tensor::Memory<bool> pointer_mask_;            // [N, max_obj]

    // Memory Encoderバッファ
    tensor::Memory<float> memory_output_;          // [1, 256, 72, 72]

    // 状態
    int current_frame_id_ = 0;
    bool is_initialized_ = false;
    bool has_memory_engines_ = false;
    std::vector<Sam3PromptUnit> registered_prompts_;
    std::pair<int, int> current_image_size_;  // width, height
};

} // namespace sam3

#endif // SAM3_TRACKER_HPP__
