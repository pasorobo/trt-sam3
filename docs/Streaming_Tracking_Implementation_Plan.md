# Streaming Tracking 実装プラン

## 概要

本ドキュメントは、`trt-sam3`プロジェクトにSAM2のStreaming Tracking機能を実装するための詳細プランです。

---

## 1. SAM2 Streaming Trackingアーキテクチャ

### 1.1 全体構成図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SAM2 Streaming Architecture                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Frame t                                                                 │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────┐                                                    │
│  │  Vision Encoder  │ ──────────────────────────────────────┐           │
│  │  (ViT + FPN)     │                                       │           │
│  └────────┬─────────┘                                       │           │
│           │                                                  │           │
│           │ Image Features                                   │           │
│           ▼                                                  │           │
│  ┌──────────────────┐      ┌──────────────────┐             │           │
│  │ Memory Attention │◄─────│   Memory Bank    │             │           │
│  │  (L=4 Blocks)    │      │  ┌────────────┐  │             │           │
│  └────────┬─────────┘      │  │Recent (N)  │  │             │           │
│           │                 │  ├────────────┤  │             │           │
│           │ Conditioned     │  │Prompted(M) │  │             │           │
│           │ Features        │  ├────────────┤  │             │           │
│           ▼                 │  │Obj Pointers│  │             │           │
│  ┌──────────────────┐      │  └────────────┘  │             │           │
│  │  Prompt Encoder  │      └──────────────────┘             │           │
│  │(Text/Geometry/   │              ▲                        │           │
│  │ Point)           │              │                        │           │
│  └────────┬─────────┘              │                        │           │
│           │                        │                        │           │
│           ▼                        │                        │           │
│  ┌──────────────────┐              │                        │           │
│  │     Decoder      │              │                        │           │
│  │ (DETR+MaskDec)   │              │                        │           │
│  └────────┬─────────┘              │                        │           │
│           │                        │                        │           │
│           │ Masks, Tokens          │                        │           │
│           ▼                        │                        │           │
│  ┌──────────────────┐              │                        │           │
│  │ Memory Encoder   │──────────────┘                        │           │
│  │ (Conv + Fusion)  │◄──────────────────────────────────────┘           │
│  └──────────────────┘     Unconditioned Image Features                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 主要コンポーネント

| コンポーネント | 役割 | 入力 | 出力 |
|--------------|------|------|------|
| **Memory Encoder** | マスク+画像特徴をメモリに変換 | マスク, 画像特徴 | メモリ特徴マップ |
| **Memory Bank** | 過去フレームのメモリを管理 | メモリ特徴 | メモリリスト |
| **Memory Attention** | 現在フレームを過去メモリで条件付け | 画像特徴, メモリ | 条件付き特徴 |
| **Object Pointers** | オブジェクトの高レベル表現 | Decoderトークン | ポインタベクトル |

---

## 2. 追加が必要なモジュール

### 2.1 Memory Encoder

#### アーキテクチャ

```
Input: pred_mask [B, H, W], uncond_features [B, C, H, W]

pred_mask
    │
    ▼
┌────────────────┐
│ Downsample     │  Conv2d(1, C, k=4, s=4)
│ (288→72)       │
└───────┬────────┘
        │ [B, C, 72, 72]
        │
        ▼
    Element-wise Add ◄── uncond_features (fpn_feat_2)
        │
        ▼
┌────────────────┐
│ Fusion Convs   │  Conv2d(C, C, k=3) × 2
│                │
└───────┬────────┘
        │
        ▼
Output: memory [B, C, 72, 72]
```

#### TensorRTエンジン仕様

| 項目 | 値 |
|-----|-----|
| 入力1 | `pred_mask`: [B, 1, 288, 288] float32 |
| 入力2 | `uncond_features`: [B, 256, 72, 72] float32 |
| 出力 | `memory`: [B, 256, 72, 72] float32 |
| 動的バッチ | 対応 (1-4) |

#### ONNXエクスポート追加コード

```python
class MemoryEncoderWrapper(nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.memory_encoder = sam2_model.memory_encoder

    def forward(self, pred_mask, uncond_features):
        # Downsample mask
        mask_downsampled = self.memory_encoder.mask_downsampler(
            pred_mask.unsqueeze(1)
        )
        # Fuse with image features
        fused = mask_downsampled + uncond_features
        # Apply fusion convs
        memory = self.memory_encoder.fuser(fused)
        return memory

def export_memory_encoder(model, output_dir, device="cuda"):
    wrapper = MemoryEncoderWrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 288, 288, device=device),
            torch.randn(1, 256, 72, 72, device=device),
        ),
        str(output_dir / "memory-encoder.onnx"),
        input_names=["pred_mask", "uncond_features"],
        output_names=["memory"],
        opset_version=17,
        dynamic_axes={
            "pred_mask": {0: "batch"},
            "uncond_features": {0: "batch"},
            "memory": {0: "batch"},
        },
    )
```

### 2.2 Memory Bank (C++実装)

#### クラス設計

```cpp
// src/infer/memory_bank.hpp
#ifndef MEMORY_BANK_HPP__
#define MEMORY_BANK_HPP__

#include <deque>
#include <vector>
#include "common/memory.hpp"

struct MemoryFrame {
    int frame_id;
    bool is_prompted;  // ユーザープロンプトありのフレームか
    std::shared_ptr<tensor::Memory<float>> spatial_memory;  // [C, H, W]
    std::shared_ptr<tensor::Memory<float>> object_pointers; // [num_objects, D]
};

class MemoryBank {
public:
    MemoryBank(int max_recent_frames = 6,
               int max_prompted_frames = 2,
               int memory_channels = 256,
               int memory_height = 72,
               int memory_width = 72);

    // メモリの追加
    void add_memory(int frame_id,
                    const tensor::Memory<float>& memory,
                    const tensor::Memory<float>& object_pointers,
                    bool is_prompted);

    // メモリの取得（Memory Attention用）
    void get_memories_for_attention(
        tensor::Memory<float>& out_memories,      // [N, C, H, W]
        tensor::Memory<float>& out_pointers,      // [N, num_obj, D]
        std::vector<int>& out_frame_ids,
        int max_frames = -1  // -1 = 全て
    ) const;

    // リセット
    void reset();

    // 状態確認
    int get_total_frames() const;
    bool has_prompted_frames() const;

private:
    int max_recent_frames_;
    int max_prompted_frames_;
    int memory_channels_;
    int memory_height_;
    int memory_width_;

    std::deque<MemoryFrame> recent_memories_;    // FIFO (最新N件)
    std::deque<MemoryFrame> prompted_memories_;  // FIFO (プロンプト付きM件)
};

#endif // MEMORY_BANK_HPP__
```

#### メモリ管理ロジック

```cpp
// src/infer/memory_bank.cpp
void MemoryBank::add_memory(int frame_id,
                            const tensor::Memory<float>& memory,
                            const tensor::Memory<float>& object_pointers,
                            bool is_prompted) {
    MemoryFrame frame;
    frame.frame_id = frame_id;
    frame.is_prompted = is_prompted;

    // GPU メモリをコピー
    frame.spatial_memory = std::make_shared<tensor::Memory<float>>();
    frame.spatial_memory->gpu(memory.gpu_bytes());
    cudaMemcpy(frame.spatial_memory->gpu(), memory.gpu(),
               memory.gpu_bytes(), cudaMemcpyDeviceToDevice);

    frame.object_pointers = std::make_shared<tensor::Memory<float>>();
    frame.object_pointers->gpu(object_pointers.gpu_bytes());
    cudaMemcpy(frame.object_pointers->gpu(), object_pointers.gpu(),
               object_pointers.gpu_bytes(), cudaMemcpyDeviceToDevice);

    if (is_prompted) {
        // プロンプト付きフレームは専用キューへ
        if (prompted_memories_.size() >= max_prompted_frames_) {
            prompted_memories_.pop_front();
        }
        prompted_memories_.push_back(std::move(frame));
    } else {
        // 通常フレームは最新Nフレームキューへ
        if (recent_memories_.size() >= max_recent_frames_) {
            recent_memories_.pop_front();
        }
        recent_memories_.push_back(std::move(frame));
    }
}
```

### 2.3 Memory Attention

#### アーキテクチャ

```
Memory Attention Block (×L=4)
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Input: current_features [B, HW, C]                         │
│         memories [N, C, H, W]                                │
│         object_pointers [N, num_obj, D]                      │
│                                                              │
│  ┌──────────────────┐                                       │
│  │ Self-Attention   │  Q, K, V from current_features        │
│  │ (with 2D RoPE)   │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Cross-Attention  │  Q from current                       │
│  │ (to memories)    │  K, V from flattened memories         │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Cross-Attention  │  Q from current                       │
│  │ (to obj ptrs)    │  K, V from object_pointers            │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │      MLP         │  Linear(C, 4C) -> GELU -> Linear(4C, C)│
│  └────────┬─────────┘                                       │
│           │                                                  │
│  Output: conditioned_features [B, HW, C]                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### TensorRTエンジン仕様

| 項目 | 値 |
|-----|-----|
| 入力1 | `current_features`: [B, 5184, 256] float32 (72×72=5184) |
| 入力2 | `memories`: [N, 256, 72, 72] float32 |
| 入力3 | `memory_mask`: [N] bool |
| 入力4 | `object_pointers`: [N, max_obj, 256] float32 |
| 入力5 | `pointer_mask`: [N, max_obj] bool |
| 出力 | `conditioned_features`: [B, 5184, 256] float32 |
| Transformer Layers | 4 |

#### ONNXエクスポート追加コード

```python
class MemoryAttentionWrapper(nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.memory_attention = sam2_model.memory_attention
        self.hidden_size = 256

    def forward(self,
                current_features,    # [B, HW, C]
                memories,            # [N, C, H, W]
                memory_mask,         # [N]
                object_pointers,     # [N, num_obj, C]
                pointer_mask):       # [N, num_obj]

        # Flatten memories: [N, C, H, W] -> [N, HW, C]
        N = memories.shape[0]
        memories_flat = memories.flatten(2).transpose(1, 2)

        # Process through L=4 attention blocks
        x = current_features
        for block in self.memory_attention.layers:
            # Self-attention on current features
            x = block.self_attn(x, x, x)
            # Cross-attention to memories
            x = block.cross_attn_memories(x, memories_flat, memories_flat)
            # Cross-attention to object pointers
            x = block.cross_attn_pointers(x, object_pointers, object_pointers)
            # MLP
            x = block.mlp(x)

        return x

def export_memory_attention(model, output_dir, device="cuda"):
    wrapper = MemoryAttentionWrapper(model).to(device).eval()

    B, N, max_obj = 1, 8, 10
    H, W, C = 72, 72, 256

    torch.onnx.export(
        wrapper,
        (
            torch.randn(B, H*W, C, device=device),
            torch.randn(N, C, H, W, device=device),
            torch.ones(N, dtype=torch.bool, device=device),
            torch.randn(N, max_obj, C, device=device),
            torch.ones(N, max_obj, dtype=torch.bool, device=device),
        ),
        str(output_dir / "memory-attention.onnx"),
        input_names=[
            "current_features", "memories", "memory_mask",
            "object_pointers", "pointer_mask"
        ],
        output_names=["conditioned_features"],
        opset_version=17,
        dynamic_axes={
            "current_features": {0: "batch"},
            "memories": {0: "num_memories"},
            "memory_mask": {0: "num_memories"},
            "object_pointers": {0: "num_memories", 1: "max_objects"},
            "pointer_mask": {0: "num_memories", 1: "max_objects"},
            "conditioned_features": {0: "batch"},
        },
    )
```

---

## 3. C++ 実装計画

### 3.1 ファイル構成（追加分）

```
src/
├── infer/
│   ├── sam3infer.hpp/cpp          # 既存（修正）
│   ├── sam3type.hpp               # 既存（修正）
│   ├── memory_bank.hpp/cpp        # 新規：メモリバンク管理
│   ├── video_predictor.hpp/cpp    # 新規：ビデオ推論クラス
│   └── infer.hpp                  # 既存（修正）
├── kernels/
│   └── memory_kernels.cu          # 新規：メモリ関連CUDAカーネル
└── interface.cpp                  # 既存（修正）
```

### 3.2 VideoPredictor クラス設計

```cpp
// src/infer/video_predictor.hpp
#ifndef VIDEO_PREDICTOR_HPP__
#define VIDEO_PREDICTOR_HPP__

#include "infer/sam3infer.hpp"
#include "infer/memory_bank.hpp"

struct VideoFrame {
    int frame_id;
    cv::Mat image;
};

struct TrackingResult {
    int frame_id;
    int object_id;
    cv::Mat mask;
    float confidence;
    std::array<float, 4> bbox;  // x1, y1, x2, y2
};

class VideoPredictor {
public:
    static std::shared_ptr<VideoPredictor> create_instance(
        const std::string& vision_encoder_path,
        const std::string& text_encoder_path,
        const std::string& geometry_encoder_path,
        const std::string& decoder_path,
        const std::string& memory_encoder_path,
        const std::string& memory_attention_path,
        int gpu_id = 0
    );

    // 初期化（最初のフレームでプロンプト指定）
    bool initialize(
        const cv::Mat& first_frame,
        const std::vector<Sam3PromptUnit>& prompts,
        float confidence_threshold = 0.5
    );

    // フレーム追加とトラッキング
    std::vector<TrackingResult> track_frame(const cv::Mat& frame);

    // 追加プロンプト（修正用）
    bool add_prompt(
        int frame_id,
        const Sam3PromptUnit& prompt
    );

    // リセット
    void reset();

    // 設定
    void set_max_objects(int max_objects);
    void set_memory_config(int max_recent, int max_prompted);

private:
    VideoPredictor(/* constructor params */);

    bool load_engines();
    void encode_and_store_memory(int frame_id, bool is_prompted);
    void apply_memory_attention();

    // エンジン
    std::shared_ptr<Sam3Infer> sam3_infer_;
    std::shared_ptr<TensorRT::Engine> memory_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> memory_attention_trt_;

    // メモリ管理
    std::unique_ptr<MemoryBank> memory_bank_;

    // 状態
    int current_frame_id_ = 0;
    bool is_initialized_ = false;

    // バッファ
    tensor::Memory<float> memory_output_;
    tensor::Memory<float> object_pointers_;
    tensor::Memory<float> conditioned_features_;

    // 設定
    int max_objects_ = 10;
    float confidence_threshold_ = 0.5;
};

#endif // VIDEO_PREDICTOR_HPP__
```

### 3.3 推論パイプライン

```cpp
// src/infer/video_predictor.cpp

std::vector<TrackingResult> VideoPredictor::track_frame(const cv::Mat& frame) {
    current_frame_id_++;
    cudaStream_t stream = nullptr;

    // Step 1: Vision Encoding
    // 既存のSam3Inferを使用
    sam3_infer_->preprocess(Sam3Input(frame), 0, stream);
    sam3_infer_->encode_image(1, stream);

    // Step 2: Memory Attention
    // 過去メモリで現在の特徴を条件付け
    if (memory_bank_->get_total_frames() > 0) {
        apply_memory_attention(stream);
    }

    // Step 3: Decode
    // 条件付き特徴を使用してマスク生成
    auto results = sam3_infer_->decode_with_conditioned_features(
        conditioned_features_,
        confidence_threshold_,
        stream
    );

    // Step 4: Memory Encoding
    // 現在フレームのメモリを生成してバンクに追加
    encode_and_store_memory(current_frame_id_, false, stream);

    // 結果を変換して返す
    return convert_to_tracking_results(results, current_frame_id_);
}

void VideoPredictor::apply_memory_attention(void* stream) {
    cudaStream_t s = (cudaStream_t)stream;

    // Memory Bankからメモリを取得
    tensor::Memory<float> memories;
    tensor::Memory<float> pointers;
    std::vector<int> frame_ids;
    memory_bank_->get_memories_for_attention(memories, pointers, frame_ids);

    int num_memories = frame_ids.size();
    if (num_memories == 0) return;

    // Memory Attention TRT 実行
    set_binding_dim(memory_attention_trt_, 0, {1, 5184, 256});
    set_binding_dim(memory_attention_trt_, 1, {num_memories, 256, 72, 72});

    memory_attention_trt_->forward({
        {"current_features", sam3_infer_->get_fpn_feat_2_flat()},
        {"memories", memories.gpu()},
        {"memory_mask", /* create mask */},
        {"object_pointers", pointers.gpu()},
        {"pointer_mask", /* create mask */},
        {"conditioned_features", conditioned_features_.gpu()}
    }, s);
}

void VideoPredictor::encode_and_store_memory(int frame_id, bool is_prompted, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;

    // Memory Encoder TRT 実行
    memory_encoder_trt_->forward({
        {"pred_mask", sam3_infer_->get_pred_masks()},
        {"uncond_features", sam3_infer_->get_fpn_feat_2()},
        {"memory", memory_output_.gpu()}
    }, s);

    // Object Pointers を Decoder トークンから抽出
    sam3_infer_->extract_object_pointers(object_pointers_);

    // Memory Bank に追加
    memory_bank_->add_memory(frame_id, memory_output_, object_pointers_, is_prompted);
}
```

---

## 4. Python API 設計

### 4.1 Python バインディング追加

```cpp
// src/interface.cpp に追加

#include "infer/video_predictor.hpp"

// VideoPredictor binding
py::class_<VideoPredictor, std::shared_ptr<VideoPredictor>>(m, "VideoPredictor")
    .def_static("create_instance", &VideoPredictor::create_instance,
        py::arg("vision_path"),
        py::arg("text_path"),
        py::arg("geometry_path"),
        py::arg("decoder_path"),
        py::arg("memory_encoder_path"),
        py::arg("memory_attention_path"),
        py::arg("gpu_id") = 0)
    .def("initialize", &VideoPredictor::initialize,
        py::arg("first_frame"),
        py::arg("prompts"),
        py::arg("confidence_threshold") = 0.5)
    .def("track_frame", &VideoPredictor::track_frame)
    .def("add_prompt", &VideoPredictor::add_prompt)
    .def("reset", &VideoPredictor::reset)
    .def("set_max_objects", &VideoPredictor::set_max_objects)
    .def("set_memory_config", &VideoPredictor::set_memory_config);

// TrackingResult binding
py::class_<TrackingResult>(m, "TrackingResult")
    .def_readonly("frame_id", &TrackingResult::frame_id)
    .def_readonly("object_id", &TrackingResult::object_id)
    .def_property_readonly("mask", [](const TrackingResult& r) {
        return mat_to_numpy(r.mask);
    })
    .def_readonly("confidence", &TrackingResult::confidence)
    .def_readonly("bbox", &TrackingResult::bbox);
```

### 4.2 Python 使用例

```python
# workspace/demo_video_tracking.py

import cv2
import numpy as np
import trtsam3

def demo_video_tracking():
    # エンジンパス
    MODEL_DIR = "engine-models"

    # VideoPredictor 初期化
    predictor = trtsam3.VideoPredictor.create_instance(
        vision_path=f"{MODEL_DIR}/vision-encoder.engine",
        text_path=f"{MODEL_DIR}/text-encoder.engine",
        geometry_path=f"{MODEL_DIR}/geometry-encoder.engine",
        decoder_path=f"{MODEL_DIR}/decoder.engine",
        memory_encoder_path=f"{MODEL_DIR}/memory-encoder.engine",
        memory_attention_path=f"{MODEL_DIR}/memory-attention.engine",
        gpu_id=0
    )

    # ビデオ読み込み
    cap = cv2.VideoCapture("input_video.mp4")
    ret, first_frame = cap.read()

    # 最初のフレームでプロンプト設定
    prompts = [
        trtsam3.Sam3PromptUnit("person"),  # テキストプロンプト
        trtsam3.Sam3PromptUnit("", [("pos", [100, 100, 200, 300])])  # ボックスプロンプト
    ]

    predictor.initialize(first_frame, prompts, confidence_threshold=0.5)

    # トラッキングループ
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # フレーム追跡
        results = predictor.track_frame(frame)

        # 可視化
        for result in results:
            mask = result.mask
            bbox = result.bbox
            obj_id = result.object_id

            # マスクオーバーレイ
            color = get_color_for_object(obj_id)
            frame[mask > 0] = frame[mask > 0] * 0.5 + np.array(color) * 0.5

            # バウンディングボックス
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{obj_id} ({result.confidence:.2f})",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_video_tracking()
```

---

## 5. 実装ロードマップ

### Phase 1: モデルエクスポート（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 1.1 | SAM2モデルのダウンロード・解析 | PyTorchモデル構造理解 |
| 1.2 | Memory Encoder ONNX エクスポート | `memory-encoder.onnx` |
| 1.3 | Memory Attention ONNX エクスポート | `memory-attention.onnx` |
| 1.4 | TensorRTエンジンビルド | `.engine` ファイル |

### Phase 2: C++コア実装（2週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 2.1 | MemoryBank クラス実装 | `memory_bank.hpp/cpp` |
| 2.2 | VideoPredictor クラス実装 | `video_predictor.hpp/cpp` |
| 2.3 | メモリ管理CUDAカーネル | `memory_kernels.cu` |
| 2.4 | Sam3Infer の修正（条件付き特徴対応） | 既存ファイル修正 |

### Phase 3: 統合・Python API（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 3.1 | pybind11 バインディング追加 | `interface.cpp` 修正 |
| 3.2 | Python デモスクリプト | `demo_video_tracking.py` |
| 3.3 | 単体テスト | テストコード |

### Phase 4: 最適化・検証（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 4.1 | パフォーマンス計測 | ベンチマーク結果 |
| 4.2 | メモリ最適化 | 最適化済みコード |
| 4.3 | ドキュメント作成 | README 更新 |

---

## 6. 推定リソース要件

### 6.1 追加VRAMメモリ

| コンポーネント | 推定サイズ | 備考 |
|---------------|-----------|------|
| Memory Encoder Engine | ~50MB | 軽量Conv |
| Memory Attention Engine | ~200MB | 4層Transformer |
| Memory Bank (8フレーム) | ~150MB | 8 × 256 × 72 × 72 × 4bytes |
| Object Pointers | ~10MB | 8 × 10 × 256 × 4bytes |
| **合計** | **~410MB** | |

### 6.2 推定推論速度

| 処理 | 推定時間 | 備考 |
|-----|---------|------|
| Vision Encoding | ~30ms | 既存と同じ |
| Memory Attention | ~10ms | 4層×CrossAttn |
| Decoding | ~10ms | 既存と同じ |
| Memory Encoding | ~5ms | 軽量Conv |
| **合計** | **~55ms/frame** | ~18 FPS |

---

## 7. リスクと対策

| リスク | 影響 | 対策 |
|-------|------|------|
| Memory Attention の動的形状 | TRT最適化困難 | 固定最大メモリ数で対応 |
| 長時間ビデオでのメモリ累積 | OOM | FIFO制限 + 定期リセット |
| オブジェクト消失時の挙動 | 追跡ロスト | Object Pointer による再検出 |
| SAM2 と SAM3 の差異 | API不整合 | 抽象レイヤーで吸収 |

---

## 8. 参考資料

- [SAM 2 公式リポジトリ](https://github.com/facebookresearch/sam2)
- [SAM 2 論文 (arXiv)](https://arxiv.org/abs/2408.00714)
- [SAM 2 アーキテクチャ解説](https://ritvik19.medium.com/papers-explained-239-sam-2-6ffb7f187281)
- [Ultralytics SAM 2 ドキュメント](https://docs.ultralytics.com/models/sam-2/)

---

*プラン作成日: 2025-12-28*
*推定実装期間: 5週間*
