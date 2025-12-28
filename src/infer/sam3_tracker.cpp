#include "infer/sam3_tracker.hpp"
#include "common/device.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace sam3 {

// ファクトリ関数
std::shared_ptr<Sam3Tracker> Sam3Tracker::create_instance(const Sam3TrackerConfig& config) {
    auto instance = std::shared_ptr<Sam3Tracker>(new Sam3Tracker(config));

    if (!instance->load_engines()) {
        std::cerr << "Failed to load Sam3Tracker engines!" << std::endl;
        return nullptr;
    }

    instance->allocate_buffers();
    return instance;
}

Sam3Tracker::Sam3Tracker(const Sam3TrackerConfig& config)
    : config_(config), current_frame_id_(0), is_initialized_(false), has_memory_engines_(false) {
    memory_bank_ = std::make_unique<MemoryBank>(config_.memory_config);
}

bool Sam3Tracker::load_engines() {
    AutoDevice device_guard(config_.gpu_id);

    // Detector (Sam3Infer) をロード
    if (!config_.geometry_encoder_path.empty()) {
        detector_ = Sam3Infer::create_instance(
            config_.vision_encoder_path,
            config_.text_encoder_path,
            config_.geometry_encoder_path,
            config_.decoder_path,
            config_.gpu_id
        );
    } else {
        detector_ = Sam3Infer::create_instance(
            config_.vision_encoder_path,
            config_.text_encoder_path,
            config_.decoder_path,
            config_.gpu_id
        );
    }

    if (!detector_) {
        std::cerr << "Failed to create detector (Sam3Infer)" << std::endl;
        return false;
    }

    // Memory Encoder をロード（オプション）
    if (!config_.memory_encoder_path.empty()) {
        memory_encoder_trt_ = TensorRT::load(config_.memory_encoder_path);
        if (!memory_encoder_trt_) {
            std::cerr << "Warning: Failed to load Memory Encoder from "
                      << config_.memory_encoder_path << std::endl;
            std::cerr << "Tracking will work in detection-only mode." << std::endl;
        }
    }

    // Memory Attention をロード（オプション）
    if (!config_.memory_attention_path.empty()) {
        memory_attention_trt_ = TensorRT::load(config_.memory_attention_path);
        if (!memory_attention_trt_) {
            std::cerr << "Warning: Failed to load Memory Attention from "
                      << config_.memory_attention_path << std::endl;
            std::cerr << "Tracking will work in detection-only mode." << std::endl;
        }
    }

    // Memory enginesが両方ロードされた場合のみ有効
    has_memory_engines_ = (memory_encoder_trt_ != nullptr) && (memory_attention_trt_ != nullptr);

    if (!has_memory_engines_) {
        std::cout << "Note: Memory engines not available. "
                  << "Running in detection-per-frame mode (no temporal tracking)." << std::endl;
    }

    return true;
}

void Sam3Tracker::allocate_buffers() {
    const auto& mem_cfg = config_.memory_config;
    int feat_size = mem_cfg.memory_channels * mem_cfg.memory_height * mem_cfg.memory_width;
    int flat_size = mem_cfg.memory_height * mem_cfg.memory_width * mem_cfg.memory_channels;

    // Vision features
    vision_features_.gpu(feat_size);
    vision_features_flat_.gpu(flat_size);

    // Memory Attention buffers
    conditioned_features_.gpu(flat_size);

    int max_memories = mem_cfg.max_recent_frames + mem_cfg.max_prompted_frames;
    gathered_memories_.gpu(max_memories * feat_size);
    gathered_pointers_.gpu(max_memories * mem_cfg.max_objects_per_frame * mem_cfg.pointer_dim);
    memory_mask_.gpu(max_memories);
    memory_mask_.cpu(max_memories);
    pointer_mask_.gpu(max_memories * mem_cfg.max_objects_per_frame);
    pointer_mask_.cpu(max_memories * mem_cfg.max_objects_per_frame);

    // Memory Encoder output
    memory_output_.gpu(feat_size);
}

bool Sam3Tracker::initialize(
    const cv::Mat& first_frame,
    const std::vector<Sam3PromptUnit>& prompts,
    float confidence_threshold) {

    if (prompts.empty()) {
        std::cerr << "Error: No prompts provided for initialization" << std::endl;
        return false;
    }

    reset();

    registered_prompts_ = prompts;
    config_.detection_threshold = confidence_threshold;
    current_image_size_ = {first_frame.cols, first_frame.rows};

    // 最初のフレームで検出を実行
    current_frame_id_ = 0;
    cudaStream_t stream = nullptr;

    // Detectorを使用して最初のフレームで検出
    std::vector<Sam3Input> inputs;
    inputs.emplace_back(first_frame, prompts, confidence_threshold);

    auto results = detector_->forwards(inputs, true, stream);

    if (!results.empty() && !results[0].empty()) {
        // 検出されたオブジェクトを登録
        for (const auto& det : results[0]) {
            TrackedObject obj;
            obj.object_id = memory_bank_->assign_new_object_id();
            obj.class_name = det->name;
            obj.last_score = det->score;
            obj.is_active = true;
            obj.last_seen_frame = current_frame_id_;

            if (det->type == object::ObjectType::Detection) {
                auto* box_det = dynamic_cast<object::DetectionBox*>(det.get());
                if (box_det) {
                    obj.last_bbox = {box_det->box.left, box_det->box.top,
                                    box_det->box.right, box_det->box.bottom};
                }
            }

            memory_bank_->register_object(obj);
        }

        // Memory encodingを実行（エンジンがある場合）
        if (has_memory_engines_) {
            encode_memory(true, stream);
        }
    }

    is_initialized_ = true;
    std::cout << "Tracker initialized with " << memory_bank_->get_active_object_ids().size()
              << " objects" << std::endl;

    return true;
}

std::vector<TrackingResult> Sam3Tracker::track_frame(const cv::Mat& frame) {
    if (!is_initialized_) {
        std::cerr << "Error: Tracker not initialized" << std::endl;
        return {};
    }

    current_frame_id_++;
    cudaStream_t stream = nullptr;
    std::vector<TrackingResult> results;

    current_image_size_ = {frame.cols, frame.rows};

    // ============================================
    // Step 1: 検出実行
    // ============================================
    std::vector<Sam3Input> inputs;
    inputs.emplace_back(frame, registered_prompts_, config_.detection_threshold);

    auto det_results = detector_->forwards(inputs, true, stream);

    // 検出結果をTrackingResultに変換
    std::vector<TrackingResult> detection_results;
    if (!det_results.empty()) {
        for (const auto& det : det_results[0]) {
            TrackingResult tr;
            tr.frame_id = current_frame_id_;
            tr.object_id = -1;  // まだ割り当てなし
            tr.class_name = det->name;
            tr.confidence = det->score;
            tr.is_new = true;

            if (det->type == object::ObjectType::Detection) {
                auto* box_det = dynamic_cast<object::DetectionBox*>(det.get());
                if (box_det) {
                    tr.bbox = {box_det->box.left, box_det->box.top,
                              box_det->box.right, box_det->box.bottom};
                }
            } else if (det->type == object::ObjectType::Segmentation) {
                auto* seg_det = dynamic_cast<object::SegmentationBox*>(det.get());
                if (seg_det) {
                    tr.bbox = {seg_det->box.left, seg_det->box.top,
                              seg_det->box.right, seg_det->box.bottom};
                    tr.mask = seg_det->seg.mask.clone();
                }
            }

            detection_results.push_back(std::move(tr));
        }
    }

    // ============================================
    // Step 2: 既存オブジェクトとのマッチング
    // ============================================
    auto active_objects = memory_bank_->get_active_objects();

    // IoUベースのマッチング
    std::vector<bool> detection_matched(detection_results.size(), false);
    std::vector<bool> object_matched(active_objects.size(), false);

    for (size_t d = 0; d < detection_results.size(); ++d) {
        float best_iou = config_.match_iou_threshold;
        int best_obj_idx = -1;

        for (size_t o = 0; o < active_objects.size(); ++o) {
            if (object_matched[o]) continue;

            // クラス名が一致するかチェック
            if (detection_results[d].class_name != active_objects[o].class_name) {
                continue;
            }

            float iou = compute_iou(detection_results[d].bbox, active_objects[o].last_bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_obj_idx = static_cast<int>(o);
            }
        }

        if (best_obj_idx >= 0) {
            // マッチ成功：既存オブジェクトに割り当て
            detection_results[d].object_id = active_objects[best_obj_idx].object_id;
            detection_results[d].is_new = false;
            detection_matched[d] = true;
            object_matched[best_obj_idx] = true;

            // オブジェクト状態を更新
            memory_bank_->update_object_state(
                detection_results[d].object_id,
                true,
                detection_results[d].confidence,
                current_frame_id_
            );

            // bboxを更新
            auto* obj = memory_bank_->get_object(detection_results[d].object_id);
            if (obj) {
                obj->last_bbox = detection_results[d].bbox;
            }
        }
    }

    // マッチしなかった検出を新規オブジェクトとして登録
    for (size_t d = 0; d < detection_results.size(); ++d) {
        if (!detection_matched[d]) {
            int new_id = memory_bank_->assign_new_object_id();
            detection_results[d].object_id = new_id;
            detection_results[d].is_new = true;

            TrackedObject obj;
            obj.object_id = new_id;
            obj.class_name = detection_results[d].class_name;
            obj.last_score = detection_results[d].confidence;
            obj.last_bbox = detection_results[d].bbox;
            obj.is_active = true;
            obj.last_seen_frame = current_frame_id_;

            memory_bank_->register_object(obj);
        }
    }

    // マッチしなかった既存オブジェクトを非アクティブ化（一定フレーム後）
    for (size_t o = 0; o < active_objects.size(); ++o) {
        if (!object_matched[o]) {
            int frames_lost = current_frame_id_ - active_objects[o].last_seen_frame;
            if (frames_lost > config_.max_lost_frames) {
                memory_bank_->deactivate_object(active_objects[o].object_id);
            }
        }
    }

    // ============================================
    // Step 3: Memory Encoding（エンジンがある場合）
    // ============================================
    if (has_memory_engines_ && !detection_results.empty()) {
        // オブジェクトポインタを抽出してメモリに追加
        auto tracked_objs = extract_object_pointers(detection_results);
        // 注: 実際のMemory Encoderがないため、簡易版として空のメモリを追加
        // Memory Encoderエンジンがロードされたら、ここで実際のエンコードを実行
    }

    // 結果を返す
    return detection_results;
}

float Sam3Tracker::compute_iou(const std::array<float, 4>& box1,
                                const std::array<float, 4>& box2) const {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - inter_area;

    if (union_area <= 0) return 0.0f;
    return inter_area / union_area;
}

std::vector<TrackedObject> Sam3Tracker::extract_object_pointers(
    const std::vector<TrackingResult>& results) const {

    std::vector<TrackedObject> objects;
    for (const auto& r : results) {
        TrackedObject obj;
        obj.object_id = r.object_id;
        obj.class_name = r.class_name;
        obj.last_score = r.confidence;
        obj.last_bbox = r.bbox;
        obj.is_active = true;
        obj.last_seen_frame = current_frame_id_;

        // ポインタは実際のDecoder出力トークンから抽出する必要がある
        // 現在は簡易版としてゼロ初期化
        obj.pointer.resize(config_.memory_config.pointer_dim, 0.0f);

        objects.push_back(std::move(obj));
    }
    return objects;
}

void Sam3Tracker::encode_memory(bool is_prompted, void* stream) {
    if (!memory_encoder_trt_) return;

    // 実際のMemory Encoder実行
    // 注: 現在はプレースホルダー。Memory Encoderエンジンの入出力仕様に合わせて実装

    // memory_encoder_trt_->forward({
    //     {"pred_mask", /* mask tensor */},
    //     {"fpn_feat_2", /* vision features */},
    //     {"memory", memory_output_.gpu()}
    // }, static_cast<cudaStream_t>(stream));
}

void Sam3Tracker::apply_memory_attention(void* stream) {
    if (!memory_attention_trt_ || memory_bank_->empty()) return;

    int num_memories = 0;
    memory_bank_->gather_for_attention(
        gathered_memories_,
        gathered_pointers_,
        memory_mask_,
        pointer_mask_,
        num_memories,
        stream
    );

    if (num_memories == 0) return;

    // 実際のMemory Attention実行
    // 注: 現在はプレースホルダー。Memory Attentionエンジンの入出力仕様に合わせて実装

    // memory_attention_trt_->forward({
    //     {"current_features", vision_features_flat_.gpu()},
    //     {"memories", gathered_memories_.gpu()},
    //     {"memory_mask", memory_mask_.gpu()},
    //     {"object_pointers", gathered_pointers_.gpu()},
    //     {"pointer_mask", pointer_mask_.gpu()},
    //     {"conditioned_features", conditioned_features_.gpu()}
    // }, static_cast<cudaStream_t>(stream));
}

bool Sam3Tracker::add_correction(int frame_id, int object_id, const Sam3PromptUnit& prompt) {
    // インタラクティブ修正の実装
    // 指定されたオブジェクトに対して追加のプロンプトを適用
    if (!memory_bank_->has_object(object_id)) {
        std::cerr << "Error: Object " << object_id << " not found" << std::endl;
        return false;
    }

    // TODO: 修正ロジックを実装
    return true;
}

bool Sam3Tracker::add_prompt(const Sam3PromptUnit& prompt) {
    // 新しいプロンプトを追加
    registered_prompts_.push_back(prompt);
    return true;
}

bool Sam3Tracker::remove_object(int object_id) {
    if (!memory_bank_->has_object(object_id)) {
        return false;
    }
    memory_bank_->deactivate_object(object_id);
    return true;
}

std::vector<int> Sam3Tracker::get_tracked_object_ids() const {
    return memory_bank_->get_active_object_ids();
}

std::vector<TrackedObject> Sam3Tracker::get_tracked_objects() const {
    return memory_bank_->get_active_objects();
}

void Sam3Tracker::reset() {
    memory_bank_->reset();
    registered_prompts_.clear();
    current_frame_id_ = 0;
    is_initialized_ = false;
}

void Sam3Tracker::set_binding_dim(std::shared_ptr<TensorRT::Engine>& engine,
                                   int idx, const std::vector<int>& dims) {
    if (engine) {
        engine->set_run_dims(idx, dims);
    }
}

} // namespace sam3
