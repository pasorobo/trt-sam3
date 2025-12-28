# SAM3 Streaming Tracking 実装プラン

## 概要

本ドキュメントは、`trt-sam3`プロジェクトにSAM3（Segment Anything Model 3）のStreaming Tracking機能を実装するための詳細プランです。

SAM3は検出器（Detector）と追跡器（Tracker）を分離した設計を採用しており、追跡器はSAM2のtransformer encoder-decoderアーキテクチャを継承しています。

---

## 1. SAM3 アーキテクチャ概要

### 1.1 SAM3 vs 現在のtrt-sam3実装

| コンポーネント | SAM3フル機能 | 現在のtrt-sam3 | 対応状況 |
|--------------|-------------|---------------|:-------:|
| Vision Encoder (共有) | ✓ | ✓ | ✅ 実装済み |
| Text Encoder | ✓ | ✓ | ✅ 実装済み |
| Geometry Encoder | ✓ | ✓ | ✅ 実装済み |
| DETR Detector | ✓ | ✓ | ✅ 実装済み |
| Mask Decoder | ✓ | ✓ | ✅ 実装済み |
| Presence Token | ✓ | ✓ | ✅ 実装済み |
| **Tracker** | ✓ | ✗ | ❌ 未実装 |
| **Memory Encoder** | ✓ | ✗ | ❌ 未実装 |
| **Memory Bank** | ✓ | ✗ | ❌ 未実装 |
| **Memory Attention** | ✓ | ✗ | ❌ 未実装 |

### 1.2 SAM3 全体構成図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAM3 Unified Architecture                            │
│                     (Detector + Tracker with Shared Vision)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Frame                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  ┌───────────────────────────────────┐                                      │
│  │     Shared Vision Encoder         │  ← Perception Encoder (Meta 2025)    │
│  │     (ViT + FPN Neck)              │     848M params total                │
│  └───────────────┬───────────────────┘                                      │
│                  │                                                           │
│         ┌───────┴───────┐                                                   │
│         │               │                                                    │
│         ▼               ▼                                                    │
│  ┌─────────────┐  ┌─────────────────────────────────────────────┐          │
│  │  DETECTOR   │  │              TRACKER                         │          │
│  │  (DETR)     │  │  (SAM2 Transformer Encoder-Decoder)         │          │
│  │             │  │                                              │          │
│  │ ┌─────────┐ │  │  ┌──────────────┐    ┌──────────────────┐  │          │
│  │ │Prompt   │ │  │  │   Memory     │◄───│   Memory Bank    │  │          │
│  │ │Encoders │ │  │  │  Attention   │    │  ┌────────────┐  │  │          │
│  │ │• Text   │ │  │  │  (L=4)       │    │  │Recent (N=6)│  │  │          │
│  │ │• Geom   │ │  │  └──────┬───────┘    │  ├────────────┤  │  │          │
│  │ │• Image  │ │  │         │            │  │Prompted(M) │  │  │          │
│  │ └────┬────┘ │  │         ▼            │  ├────────────┤  │  │          │
│  │      │      │  │  ┌──────────────┐    │  │Obj Pointers│  │  │          │
│  │      ▼      │  │  │  Transformer │    │  └────────────┘  │  │          │
│  │ ┌─────────┐ │  │  │  Decoder     │    └─────────▲────────┘  │          │
│  │ │DETR     │ │  │  └──────┬───────┘              │           │          │
│  │ │Encoder  │ │  │         │                      │           │          │
│  │ │Decoder  │ │  │         ▼                      │           │          │
│  │ └────┬────┘ │  │  ┌──────────────┐              │           │          │
│  │      │      │  │  │ Mask Decoder │              │           │          │
│  │      ▼      │  │  └──────┬───────┘              │           │          │
│  │ ┌─────────┐ │  │         │                      │           │          │
│  │ │ Mask    │ │  │         ▼                      │           │          │
│  │ │Decoder  │ │  │  ┌──────────────┐              │           │          │
│  │ └────┬────┘ │  │  │   Memory     │──────────────┘           │          │
│  │      │      │  │  │  Encoder     │                          │          │
│  └──────┼──────┘  │  └──────────────┘                          │          │
│         │         └─────────────────────────────────────────────┘          │
│         │                          │                                        │
│         ▼                          ▼                                        │
│  ┌─────────────┐           ┌─────────────┐                                 │
│  │  Detection  │           │  Tracking   │                                 │
│  │  Results    │           │  Results    │                                 │
│  │  (per-frame)│           │  (temporal) │                                 │
│  └─────────────┘           └─────────────┘                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │               Matching & Update Stage                                │  │
│  │   • Merge propagated masklets with newly detected masks              │  │
│  │   • Maintain temporal consistency under occlusion/reappearance       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 SAM3 Trackerの特徴

SAM3のTrackerは以下の特徴を持ちます：

| 特徴 | 説明 |
|-----|------|
| **分離設計** | DetectorとTrackerが分離され、タスク干渉を最小化 |
| **共有ビジョン** | Vision Encoderを共有し、計算効率を向上 |
| **SAM2継承** | SAM2のtransformer encoder-decoder + memory機構を継承 |
| **Presence Token** | テキストプロンプトの識別性向上（例：「白い服のプレイヤー」vs「赤い服のプレイヤー」） |
| **Masklet Propagation** | 時間的オブジェクトセグメントの伝播 |

---

## 2. 追加が必要なモジュール

### 2.1 SAM3 Tracker モジュール構成

```
SAM3 Tracker
├── Memory Encoder          # マスク+特徴 → メモリ
├── Memory Bank             # 時間的メモリ管理
│   ├── Recent Frame Cache  # 最新Nフレーム
│   ├── Prompted Frames     # プロンプト付きフレーム
│   └── Object Pointers     # オブジェクト表現
├── Memory Attention        # 現在フレームの条件付け
│   ├── Self-Attention      # 空間関係
│   ├── Cross-Attn (Memory) # 時間的コンテキスト
│   └── Cross-Attn (Pointer)# オブジェクト情報
├── Transformer Decoder     # マスク生成
└── Matching & Update       # 検出・追跡結果の統合
```

### 2.2 Memory Encoder

#### アーキテクチャ詳細

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAM3 Memory Encoder                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Inputs:                                                         │
│    • pred_mask: [B, 1, 288, 288] - Decoder出力マスク            │
│    • fpn_feat_2: [B, 256, 72, 72] - Vision特徴（非条件付き）    │
│    • obj_score: [B, num_queries] - オブジェクトスコア           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  pred_mask ────► Mask Downsampler ────┐                  │   │
│  │                  (Conv2d k=4, s=4)    │                  │   │
│  │                  [B,1,288,288]→[B,256,72,72]             │   │
│  │                                        │                  │   │
│  │  fpn_feat_2 ──────────────────────────┼─► Element-wise  │   │
│  │                                        │      Add        │   │
│  │                                        │        │        │   │
│  │                                        ▼        ▼        │   │
│  │                               ┌──────────────────────┐   │   │
│  │                               │   Fusion Module      │   │   │
│  │                               │   (Conv×2 + LN)      │   │   │
│  │                               └──────────┬───────────┘   │   │
│  │                                          │               │   │
│  │                                          ▼               │   │
│  │                               ┌──────────────────────┐   │   │
│  │                               │  Spatial Memory      │   │   │
│  │                               │  [B, 256, 72, 72]    │   │   │
│  │                               └──────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Output:                                                         │
│    • memory: [B, 256, 72, 72] - フレームメモリ                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### TensorRTエンジン仕様

| 項目 | 値 |
|-----|-----|
| 入力1 | `pred_mask`: [B, 1, 288, 288] float32 |
| 入力2 | `fpn_feat_2`: [B, 256, 72, 72] float32 |
| 出力 | `memory`: [B, 256, 72, 72] float32 |
| 動的バッチ | 対応 (1-4) |
| 推定サイズ | ~50MB |

#### ONNXエクスポートコード

```python
# workspace/script/export_tracker.py

class Sam3MemoryEncoderWrapper(nn.Module):
    """SAM3 Memory Encoder for TensorRT export"""

    def __init__(self, sam3_tracker_model):
        super().__init__()
        self.mask_downsampler = sam3_tracker_model.memory_encoder.mask_downsampler
        self.fuser = sam3_tracker_model.memory_encoder.fuser

    def forward(self, pred_mask: torch.Tensor,
                fpn_feat_2: torch.Tensor) -> torch.Tensor:
        # Downsample mask to match feature resolution
        # [B, 1, 288, 288] -> [B, 256, 72, 72]
        mask_feat = self.mask_downsampler(pred_mask)

        # Fuse with image features
        fused = mask_feat + fpn_feat_2

        # Apply fusion convolutions
        memory = self.fuser(fused)

        return memory


def export_memory_encoder(tracker_model, output_dir: Path, device: str = "cuda"):
    print("Exporting SAM3 Memory Encoder...")
    wrapper = Sam3MemoryEncoderWrapper(tracker_model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 1, 288, 288, device=device),
            torch.randn(1, 256, 72, 72, device=device),
        ),
        str(output_dir / "memory-encoder.onnx"),
        input_names=["pred_mask", "fpn_feat_2"],
        output_names=["memory"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "pred_mask": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "memory": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'memory-encoder.onnx'}")
```

### 2.3 Memory Attention

#### アーキテクチャ詳細

```
┌─────────────────────────────────────────────────────────────────┐
│               SAM3 Memory Attention (L=4 Layers)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Inputs:                                                         │
│    • current_feat: [B, 5184, 256] - 現在フレーム特徴            │
│    • memories: [N, 256, 72, 72] - 過去フレームメモリ            │
│    • object_ptrs: [N, max_obj, 256] - オブジェクトポインタ      │
│    • temporal_pos: [N] - フレーム位置エンコーディング           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Layer 1-4 (Repeated)                                      │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  Self-Attention (with 2D RoPE)                     │   │ │
│  │  │  • Q, K, V from current_feat                       │   │ │
│  │  │  • Spatial relationships within current frame      │   │ │
│  │  └─────────────────────┬──────────────────────────────┘   │ │
│  │                        │                                   │ │
│  │                        ▼                                   │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  Cross-Attention to Memories                       │   │ │
│  │  │  • Q from current_feat                             │   │ │
│  │  │  • K, V from flattened memories [N*5184, 256]     │   │ │
│  │  │  • Temporal context from previous frames          │   │ │
│  │  └─────────────────────┬──────────────────────────────┘   │ │
│  │                        │                                   │ │
│  │                        ▼                                   │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  Cross-Attention to Object Pointers               │   │ │
│  │  │  • Q from current_feat                             │   │ │
│  │  │  • K, V from object_ptrs [N*max_obj, 256]         │   │ │
│  │  │  • High-level semantic object information         │   │ │
│  │  └─────────────────────┬──────────────────────────────┘   │ │
│  │                        │                                   │ │
│  │                        ▼                                   │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  FFN (Feed-Forward Network)                        │   │ │
│  │  │  Linear(256, 1024) → GELU → Linear(1024, 256)     │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output:                                                         │
│    • conditioned_feat: [B, 5184, 256] - 条件付き特徴            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### TensorRTエンジン仕様

| 項目 | 値 |
|-----|-----|
| 入力1 | `current_features`: [B, 5184, 256] float32 |
| 入力2 | `memories`: [N, 256, 72, 72] float32 |
| 入力3 | `memory_mask`: [N] bool |
| 入力4 | `object_pointers`: [N, max_obj, 256] float32 |
| 入力5 | `pointer_mask`: [N, max_obj] bool |
| 出力 | `conditioned_features`: [B, 5184, 256] float32 |
| Transformer Layers | 4 |
| 推定サイズ | ~200MB |

#### ONNXエクスポートコード

```python
class Sam3MemoryAttentionWrapper(nn.Module):
    """SAM3 Memory Attention for TensorRT export"""

    def __init__(self, sam3_tracker_model):
        super().__init__()
        self.memory_attention = sam3_tracker_model.memory_attention
        self.num_layers = 4

    def forward(self,
                current_features: torch.Tensor,    # [B, HW, C]
                memories: torch.Tensor,            # [N, C, H, W]
                memory_mask: torch.Tensor,         # [N]
                object_pointers: torch.Tensor,     # [N, max_obj, C]
                pointer_mask: torch.Tensor         # [N, max_obj]
                ) -> torch.Tensor:

        N, C, H, W = memories.shape
        # Flatten memories: [N, C, H, W] -> [N*HW, C]
        memories_flat = memories.flatten(2).transpose(1, 2).reshape(-1, C)

        # Flatten object pointers: [N, max_obj, C] -> [N*max_obj, C]
        pointers_flat = object_pointers.reshape(-1, C)

        x = current_features
        for layer in self.memory_attention.layers:
            # Self-attention with 2D RoPE
            x = layer.self_attn(x)
            # Cross-attention to memories
            x = layer.cross_attn_mem(x, memories_flat, memory_mask)
            # Cross-attention to object pointers
            x = layer.cross_attn_ptr(x, pointers_flat, pointer_mask)
            # FFN
            x = layer.ffn(x)

        return x


def export_memory_attention(tracker_model, output_dir: Path, device: str = "cuda"):
    print("Exporting SAM3 Memory Attention...")
    wrapper = Sam3MemoryAttentionWrapper(tracker_model).to(device).eval()

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
        do_constant_folding=True,
        dynamic_axes={
            "current_features": {0: "batch"},
            "memories": {0: "num_memories"},
            "memory_mask": {0: "num_memories"},
            "object_pointers": {0: "num_memories", 1: "max_objects"},
            "pointer_mask": {0: "num_memories", 1: "max_objects"},
            "conditioned_features": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'memory-attention.onnx'}")
```

---

## 3. C++ 実装計画

### 3.1 ファイル構成

```
src/
├── infer/
│   ├── sam3infer.hpp/cpp          # 既存（修正：Tracker連携追加）
│   ├── sam3type.hpp               # 既存（修正：Tracking型追加）
│   ├── sam3_tracker.hpp/cpp       # 新規：SAM3 Trackerクラス
│   ├── memory_bank.hpp/cpp        # 新規：Memory Bank管理
│   ├── object_tracker.hpp/cpp     # 新規：オブジェクトID管理
│   └── infer.hpp                  # 既存（修正）
├── kernels/
│   ├── preprocess.cu              # 既存
│   ├── postprocess.cu             # 既存
│   └── tracking_kernels.cu        # 新規：追跡用CUDAカーネル
└── interface.cpp                  # 既存（修正：Tracker API追加）
```

### 3.2 Memory Bank クラス

```cpp
// src/infer/memory_bank.hpp
#ifndef MEMORY_BANK_HPP__
#define MEMORY_BANK_HPP__

#include <deque>
#include <vector>
#include <unordered_map>
#include "common/memory.hpp"

// 単一オブジェクトの追跡情報
struct TrackedObject {
    int object_id;
    std::string class_name;
    float last_score;
    std::array<float, 256> pointer;  // オブジェクトポインタ
    bool is_active;
};

// フレームメモリ
struct FrameMemory {
    int frame_id;
    bool is_prompted;
    std::shared_ptr<tensor::Memory<float>> spatial_memory;  // [256, 72, 72]
    std::vector<TrackedObject> objects;
};

class MemoryBank {
public:
    struct Config {
        int max_recent_frames = 6;      // 最新フレームキャッシュ数
        int max_prompted_frames = 2;    // プロンプト付きフレーム数
        int max_objects_per_frame = 10; // フレームあたり最大オブジェクト数
        int memory_channels = 256;
        int memory_size = 72;           // 72x72
    };

    explicit MemoryBank(const Config& config = Config());

    // メモリ追加
    void add_frame_memory(
        int frame_id,
        const tensor::Memory<float>& spatial_memory,
        const std::vector<TrackedObject>& objects,
        bool is_prompted
    );

    // Memory Attention用にメモリを取得
    void gather_for_attention(
        tensor::Memory<float>& out_memories,      // [N, 256, 72, 72]
        tensor::Memory<float>& out_pointers,      // [N, max_obj, 256]
        tensor::Memory<bool>& out_memory_mask,    // [N]
        tensor::Memory<bool>& out_pointer_mask,   // [N, max_obj]
        int& num_memories
    ) const;

    // オブジェクト追跡状態管理
    void update_object_state(int object_id, bool is_active, float score);
    std::vector<int> get_active_object_ids() const;

    // リセット
    void reset();
    void clear_recent();

    // 状態確認
    int get_frame_count() const { return recent_.size() + prompted_.size(); }
    bool empty() const { return recent_.empty() && prompted_.empty(); }

private:
    Config config_;
    std::deque<FrameMemory> recent_;     // 最新フレーム（FIFO）
    std::deque<FrameMemory> prompted_;   // プロンプト付きフレーム
    std::unordered_map<int, TrackedObject> active_objects_;  // アクティブオブジェクト

    int next_object_id_ = 0;
};

#endif // MEMORY_BANK_HPP__
```

### 3.3 SAM3 Tracker クラス

```cpp
// src/infer/sam3_tracker.hpp
#ifndef SAM3_TRACKER_HPP__
#define SAM3_TRACKER_HPP__

#include "infer/sam3infer.hpp"
#include "infer/memory_bank.hpp"
#include "common/tensorrt.hpp"
#include <memory>

// 追跡結果
struct TrackingResult {
    int frame_id;
    int object_id;
    std::string class_name;
    float confidence;
    std::array<float, 4> bbox;  // x1, y1, x2, y2
    cv::Mat mask;               // セグメンテーションマスク
    bool is_new;                // 新規検出か追跡継続か
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

    // Memory Bank設定
    MemoryBank::Config memory_config;
};

class Sam3Tracker {
public:
    static std::shared_ptr<Sam3Tracker> create_instance(const Sam3TrackerConfig& config);

    // 初期化（最初のフレームでプロンプト設定）
    bool initialize(
        const cv::Mat& first_frame,
        const std::vector<Sam3PromptUnit>& prompts
    );

    // ビデオフレーム追跡
    std::vector<TrackingResult> track_frame(const cv::Mat& frame);

    // インタラクティブ修正
    bool add_correction(
        int frame_id,
        int object_id,
        const Sam3PromptUnit& prompt
    );

    // オブジェクト管理
    bool remove_object(int object_id);
    std::vector<int> get_tracked_objects() const;

    // リセット
    void reset();

    // 状態取得
    int get_current_frame_id() const { return current_frame_id_; }
    bool is_initialized() const { return is_initialized_; }

private:
    Sam3Tracker(const Sam3TrackerConfig& config);

    bool load_engines();
    void allocate_buffers();

    // 推論パイプライン
    void encode_frame(const cv::Mat& frame, void* stream);
    void apply_memory_attention(void* stream);
    void run_detection(const std::vector<Sam3PromptUnit>& prompts, void* stream);
    void run_tracking_propagation(void* stream);
    void encode_memory(bool is_prompted, void* stream);
    void match_and_update(
        std::vector<TrackingResult>& detection_results,
        std::vector<TrackingResult>& tracking_results
    );

    // 設定
    Sam3TrackerConfig config_;

    // エンジン
    std::shared_ptr<Sam3Infer> detector_;  // 既存のDetector
    std::shared_ptr<TensorRT::Engine> memory_encoder_trt_;
    std::shared_ptr<TensorRT::Engine> memory_attention_trt_;

    // メモリ管理
    std::unique_ptr<MemoryBank> memory_bank_;

    // バッファ
    tensor::Memory<float> current_features_;      // [1, 5184, 256]
    tensor::Memory<float> conditioned_features_;  // [1, 5184, 256]
    tensor::Memory<float> memory_output_;         // [1, 256, 72, 72]
    tensor::Memory<float> gathered_memories_;     // [N, 256, 72, 72]
    tensor::Memory<float> gathered_pointers_;     // [N, max_obj, 256]
    tensor::Memory<bool> memory_mask_;
    tensor::Memory<bool> pointer_mask_;

    // 状態
    int current_frame_id_ = 0;
    bool is_initialized_ = false;
    std::vector<Sam3PromptUnit> registered_prompts_;
};

#endif // SAM3_TRACKER_HPP__
```

### 3.4 推論パイプライン実装

```cpp
// src/infer/sam3_tracker.cpp

std::vector<TrackingResult> Sam3Tracker::track_frame(const cv::Mat& frame) {
    current_frame_id_++;
    cudaStream_t stream = nullptr;
    std::vector<TrackingResult> results;

    // ============================================
    // Step 1: Vision Encoding（共有）
    // ============================================
    encode_frame(frame, stream);

    // ============================================
    // Step 2: Detection（新規オブジェクト検出）
    // ============================================
    std::vector<TrackingResult> detection_results;
    if (!registered_prompts_.empty()) {
        run_detection(registered_prompts_, stream);
        detection_results = collect_detection_results(config_.detection_threshold);
    }

    // ============================================
    // Step 3: Tracking（Memory-based Propagation）
    // ============================================
    std::vector<TrackingResult> tracking_results;
    if (!memory_bank_->empty()) {
        // Memory Attentionで過去コンテキストを適用
        apply_memory_attention(stream);

        // Trackingデコード
        run_tracking_propagation(stream);
        tracking_results = collect_tracking_results(config_.tracking_threshold);
    }

    // ============================================
    // Step 4: Matching & Update
    // ============================================
    match_and_update(detection_results, tracking_results);

    // 結果を統合
    for (auto& det : detection_results) {
        if (!det.is_new) continue;  // マッチしなかった新規検出のみ
        det.object_id = memory_bank_->assign_new_object_id();
        results.push_back(std::move(det));
    }
    for (auto& trk : tracking_results) {
        results.push_back(std::move(trk));
    }

    // ============================================
    // Step 5: Memory Encoding & Storage
    // ============================================
    encode_memory(false, stream);

    return results;
}

void Sam3Tracker::apply_memory_attention(void* stream) {
    cudaStream_t s = (cudaStream_t)stream;

    // Memory Bankからメモリを取得
    int num_memories = 0;
    memory_bank_->gather_for_attention(
        gathered_memories_,
        gathered_pointers_,
        memory_mask_,
        pointer_mask_,
        num_memories
    );

    if (num_memories == 0) {
        // メモリなしの場合、現在の特徴をそのまま使用
        cudaMemcpyAsync(
            conditioned_features_.gpu(),
            current_features_.gpu(),
            current_features_.gpu_bytes(),
            cudaMemcpyDeviceToDevice, s
        );
        return;
    }

    // Memory Attention実行
    memory_attention_trt_->set_run_dims(0, {1, 5184, 256});
    memory_attention_trt_->set_run_dims(1, {num_memories, 256, 72, 72});

    memory_attention_trt_->forward({
        {"current_features", current_features_.gpu()},
        {"memories", gathered_memories_.gpu()},
        {"memory_mask", memory_mask_.gpu()},
        {"object_pointers", gathered_pointers_.gpu()},
        {"pointer_mask", pointer_mask_.gpu()},
        {"conditioned_features", conditioned_features_.gpu()}
    }, s);
}

void Sam3Tracker::encode_memory(bool is_prompted, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;

    // Memory Encoder実行
    memory_encoder_trt_->forward({
        {"pred_mask", detector_->get_output_mask()},
        {"fpn_feat_2", detector_->get_fpn_feat_2()},
        {"memory", memory_output_.gpu()}
    }, s);

    // オブジェクトポインタを抽出
    auto objects = extract_tracked_objects();

    // Memory Bankに追加
    memory_bank_->add_frame_memory(
        current_frame_id_,
        memory_output_,
        objects,
        is_prompted
    );
}
```

---

## 4. Python API

### 4.1 pybind11バインディング

```cpp
// src/interface.cpp に追加

#include "infer/sam3_tracker.hpp"

PYBIND11_MODULE(trtsam3, m) {
    // 既存のバインディング...

    // === SAM3 Tracker バインディング ===

    py::class_<TrackingResult>(m, "TrackingResult")
        .def_readonly("frame_id", &TrackingResult::frame_id)
        .def_readonly("object_id", &TrackingResult::object_id)
        .def_readonly("class_name", &TrackingResult::class_name)
        .def_readonly("confidence", &TrackingResult::confidence)
        .def_readonly("bbox", &TrackingResult::bbox)
        .def_property_readonly("mask", [](const TrackingResult& r) {
            return mat_to_numpy(r.mask);
        })
        .def_readonly("is_new", &TrackingResult::is_new)
        .def("__repr__", [](const TrackingResult& r) {
            return "TrackingResult(obj_id=" + std::to_string(r.object_id) +
                   ", class=" + r.class_name +
                   ", conf=" + std::to_string(r.confidence) + ")";
        });

    py::class_<Sam3TrackerConfig>(m, "Sam3TrackerConfig")
        .def(py::init<>())
        .def_readwrite("vision_encoder_path", &Sam3TrackerConfig::vision_encoder_path)
        .def_readwrite("text_encoder_path", &Sam3TrackerConfig::text_encoder_path)
        .def_readwrite("geometry_encoder_path", &Sam3TrackerConfig::geometry_encoder_path)
        .def_readwrite("decoder_path", &Sam3TrackerConfig::decoder_path)
        .def_readwrite("memory_encoder_path", &Sam3TrackerConfig::memory_encoder_path)
        .def_readwrite("memory_attention_path", &Sam3TrackerConfig::memory_attention_path)
        .def_readwrite("gpu_id", &Sam3TrackerConfig::gpu_id)
        .def_readwrite("detection_threshold", &Sam3TrackerConfig::detection_threshold)
        .def_readwrite("tracking_threshold", &Sam3TrackerConfig::tracking_threshold);

    py::class_<Sam3Tracker, std::shared_ptr<Sam3Tracker>>(m, "Sam3Tracker")
        .def_static("create", &Sam3Tracker::create_instance)
        .def("initialize", &Sam3Tracker::initialize,
            py::arg("first_frame"),
            py::arg("prompts"))
        .def("track_frame", &Sam3Tracker::track_frame,
            py::arg("frame"))
        .def("add_correction", &Sam3Tracker::add_correction,
            py::arg("frame_id"),
            py::arg("object_id"),
            py::arg("prompt"))
        .def("remove_object", &Sam3Tracker::remove_object)
        .def("get_tracked_objects", &Sam3Tracker::get_tracked_objects)
        .def("reset", &Sam3Tracker::reset)
        .def_property_readonly("current_frame_id", &Sam3Tracker::get_current_frame_id)
        .def_property_readonly("is_initialized", &Sam3Tracker::is_initialized);
}
```

### 4.2 Python使用例

```python
# workspace/demo_sam3_tracking.py

import cv2
import numpy as np
import trtsam3

def get_color_for_object(obj_id: int) -> tuple:
    """オブジェクトIDに基づいて色を生成"""
    np.random.seed(obj_id * 123)
    return tuple(np.random.randint(50, 255, 3).tolist())

def demo_sam3_video_tracking():
    """SAM3 Streaming Tracking デモ"""

    # 設定
    config = trtsam3.Sam3TrackerConfig()
    config.vision_encoder_path = "engine-models/vision-encoder.engine"
    config.text_encoder_path = "engine-models/text-encoder.engine"
    config.geometry_encoder_path = "engine-models/geometry-encoder.engine"
    config.decoder_path = "engine-models/decoder.engine"
    config.memory_encoder_path = "engine-models/memory-encoder.engine"
    config.memory_attention_path = "engine-models/memory-attention.engine"
    config.gpu_id = 0
    config.detection_threshold = 0.5
    config.tracking_threshold = 0.3

    # Tracker作成
    tracker = trtsam3.Sam3Tracker.create(config)

    # ビデオ読み込み
    video_path = "input_video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 出力ビデオ設定
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_tracking.mp4", fourcc, fps, (width, height))

    # 最初のフレームを読み込み
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # プロンプト設定（テキストまたはボックス）
    prompts = [
        trtsam3.Sam3PromptUnit("person"),   # 人物を追跡
        trtsam3.Sam3PromptUnit("car"),      # 車を追跡
    ]

    # 初期化
    print("Initializing tracker with prompts...")
    tracker.initialize(first_frame, prompts)

    # 追跡ループ
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 追跡実行
        results = tracker.track_frame(frame)

        # 可視化
        vis_frame = frame.copy()
        for result in results:
            color = get_color_for_object(result.object_id)

            # マスクオーバーレイ
            if result.mask is not None and result.mask.size > 0:
                mask_bool = result.mask > 0
                overlay = vis_frame.copy()
                overlay[mask_bool] = color
                vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)

            # バウンディングボックス
            x1, y1, x2, y2 = map(int, result.bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # ラベル
            label = f"ID:{result.object_id} {result.class_name} {result.confidence:.2f}"
            if result.is_new:
                label += " [NEW]"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # フレーム情報
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Objects: {len(results)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 出力
        out.write(vis_frame)
        cv2.imshow("SAM3 Tracking", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Processed {frame_count} frames.")

if __name__ == "__main__":
    demo_sam3_video_tracking()
```

---

## 5. 実装ロードマップ

### Phase 1: SAM3 Trackerモデルのエクスポート（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 1.1 | SAM3 Trackerモデルのダウンロード・解析 | モデル構造理解 |
| 1.2 | HuggingFace transformersからSam3TrackerModel取得 | PyTorchモデル |
| 1.3 | Memory Encoder ONNXエクスポート | `memory-encoder.onnx` |
| 1.4 | Memory Attention ONNXエクスポート | `memory-attention.onnx` |
| 1.5 | TensorRTエンジンビルド | `.engine` ファイル |

### Phase 2: C++コア実装（2週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 2.1 | MemoryBank クラス実装 | `memory_bank.hpp/cpp` |
| 2.2 | Sam3Tracker クラス実装 | `sam3_tracker.hpp/cpp` |
| 2.3 | 追跡用CUDAカーネル（IoU計算等） | `tracking_kernels.cu` |
| 2.4 | Sam3Inferの拡張（Tracker連携） | 既存ファイル修正 |
| 2.5 | オブジェクトマッチング実装 | ハンガリアンアルゴリズム |

### Phase 3: 統合・Python API（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 3.1 | pybind11バインディング追加 | `interface.cpp` 修正 |
| 3.2 | Pythonデモスクリプト | `demo_sam3_tracking.py` |
| 3.3 | 単体テスト | テストコード |

### Phase 4: 最適化・検証（1週間）

| タスク | 詳細 | 成果物 |
|-------|------|--------|
| 4.1 | パフォーマンス計測・最適化 | ベンチマーク結果 |
| 4.2 | 長時間ビデオでの安定性テスト | テスト結果 |
| 4.3 | ドキュメント更新 | README更新 |

---

## 6. 推定リソース要件

### 6.1 追加VRAMメモリ

| コンポーネント | 推定サイズ | 計算根拠 |
|---------------|-----------|---------|
| Memory Encoder Engine | ~50MB | 軽量Conv層 |
| Memory Attention Engine | ~200MB | 4層Transformer |
| Memory Bank (8フレーム) | ~150MB | 8 × 256 × 72 × 72 × 4 bytes |
| Object Pointers (80個) | ~8MB | 8 × 10 × 256 × 4 bytes |
| 処理バッファ | ~100MB | 中間特徴等 |
| **合計** | **~510MB** | |

### 6.2 推定推論速度

| 処理 | 推定時間 | 備考 |
|-----|---------|------|
| Vision Encoding | ~30ms | 既存（共有） |
| Memory Attention | ~10ms | 4層CrossAttn |
| Detection Decoding | ~10ms | DETR Decoder |
| Tracking Propagation | ~5ms | 軽量 |
| Memory Encoding | ~5ms | 軽量Conv |
| Matching & Update | ~2ms | CPU処理 |
| **合計** | **~62ms/frame** | **~16 FPS** |

### 6.3 既存実装との比較

| 項目 | 既存（Detection Only） | 追加後（Tracking） | 差分 |
|-----|----------------------|-------------------|------|
| VRAM | ~2GB | ~2.5GB | +510MB |
| 推論時間 | ~50ms | ~62ms | +12ms |
| FPS | ~20 | ~16 | -4 |

---

## 7. リスクと対策

| リスク | 影響度 | 対策 |
|-------|:------:|------|
| SAM3 Trackerモデルの入手困難 | 高 | HuggingFace transformersの最新版確認、公式リポジトリからの取得 |
| Memory Attentionの動的形状 | 中 | 最大メモリ数を固定、パディング使用 |
| 長時間ビデオでのドリフト | 中 | 定期的なリセット機構、信頼度ベースのメモリ更新 |
| オクルージョン時の追跡失敗 | 中 | Object Pointerによる再識別、IoUベースのマッチング改善 |
| GPUメモリ不足 | 低 | Memory Bankサイズの動的調整、FP16使用 |

---

## 8. 参考資料

- [SAM 3 公式リポジトリ](https://github.com/facebookresearch/sam3)
- [SAM 3 論文](https://arxiv.org/abs/2511.16719)
- [SAM 3 Tracker (HuggingFace)](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_tracker)
- [Meta AI SAM 3 公式ページ](https://ai.meta.com/sam3/)
- [Ultralytics SAM 3 ドキュメント](https://docs.ultralytics.com/models/sam-3/)

---

*プラン作成日: 2025-12-28*
*推定実装期間: 5週間*
*対象モデル: SAM3 (Segment Anything Model 3)*
