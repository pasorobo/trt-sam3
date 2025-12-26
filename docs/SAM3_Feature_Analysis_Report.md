# TRT-SAM3 機能分析レポート

## 概要

本レポートは、`trt-sam3`プロジェクトのSAM3（Segment Anything Model 3）機能の実装状況を分析し、特にStreaming Tracking対応状況と高速化効果について定量的に報告します。

---

## 1. プロジェクト構成

### 1.1 アーキテクチャ概要

```
trt-sam3/
├── src/                      # C++/CUDAソースコード
│   ├── main.cpp             # メインエントリーポイント
│   ├── interface.cpp        # Python バインディング (pybind11)
│   ├── infer/               # 推論エンジン
│   │   ├── sam3infer.hpp/cpp   # コア推論クラス
│   │   ├── sam3type.hpp        # データ型定義
│   │   └── infer.hpp           # 抽象基底クラス
│   ├── kernels/             # CUDA カーネル
│   ├── common/              # 共通ユーティリティ
│   └── osd/                 # 可視化モジュール
├── workspace/               # ランタイム環境
│   ├── demo.py              # Pythonデモ
│   ├── script/              # ユーティリティスクリプト
│   └── engine-models/       # TensorRTエンジンファイル
└── docs/                    # ドキュメント
```

### 1.2 モデル構成

本プロジェクトは4つのTensorRTエンジンで構成されています：

| エンジン | ファイル名 | 役割 |
|---------|-----------|------|
| Vision Encoder | `vision-encoder.engine` | ViTバックボーン + FPNネック。画像を特徴マップに変換 |
| Text Encoder | `text-encoder.engine` | CLIPテキストエンコーダー。テキストプロンプトを埋め込みに変換 |
| Geometry Encoder | `geometry-encoder.engine` | ボックスプロンプトを埋め込みに変換 |
| Decoder | `decoder.engine` | DETR Encoder/Decoder + Mask Decoder。セグメンテーションマスク生成 |

---

## 2. SAM3機能の実装状況

### 2.1 対応機能一覧

| 機能 | 対応状況 | 備考 |
|-----|:-------:|------|
| 画像セグメンテーション | ✅ 対応 | 単一画像のセグメンテーションに完全対応 |
| テキストプロンプト | ✅ 対応 | CLIPトークナイザーによるテキスト認識 |
| ボックスプロンプト（Geometry） | ✅ 対応 | 正例(pos)/負例(neg)のボックス指定に対応 |
| 混合プロンプト | ✅ 対応 | テキスト+ボックスの組み合わせに対応 |
| マルチクラス同時認識 | ✅ 対応 | 複数のテキストプロンプトを同時に処理可能 |
| クロスイメージ推論 | ✅ 対応 | A画像でプロンプト設定、B画像で認識 |
| バッチ処理 | ✅ 対応 | 最大2画像/バッチ、最大4プロンプト/デコードバッチ |
| **ビデオセグメンテーション** | ❌ 未対応 | Memory Bank未実装 |
| **Streaming Tracking** | ❌ 未対応 | 時間的追跡機能未実装 |
| ポイントプロンプト | ❌ 未対応 | ボックスプロンプトのみ対応 |

### 2.2 実装済み機能の詳細

#### テキストプロンプト
```python
# 使用例 (demo.py)
prompts_text = ["person", "head", "stair", "clothes"]
register_prompts(engine, tokenizer, prompts_text)
```
- CLIPトークナイザーを使用（シーケンス長: 32）
- 事前登録によるトークンキャッシュに対応

#### ボックスプロンプト
```python
# 使用例
box_prompt = ("pos", [x1, y1, x2, y2])  # 正例ボックス
box_prompt = ("neg", [x1, y1, x2, y2])  # 負例ボックス（除外領域）
```
- 最大20ボックス/プロンプトに対応
- 正規化座標（0-1）への自動変換

#### クロスイメージ推論（A→B推論）
```python
# A画像でプロンプトを設定し、B画像で認識
engine.setup_geometry_input(prompt_image, "label", [box_prompt])
results = engine.forwards([input_obj], "label", True)
```
- Geometry特徴のキャッシュ機能
- ラベルベースの特徴管理

---

## 3. Streaming Tracking 対応状況

### 3.1 結論

**Streaming Tracking は未対応です。**

### 3.2 未対応の理由

SAM2/SAM3のStreaming Tracking機能には以下のコンポーネントが必要ですが、本プロジェクトでは実装されていません：

| コンポーネント | 役割 | 実装状況 |
|--------------|------|:-------:|
| Memory Encoder | フレーム特徴をメモリ表現に変換 | ❌ 未実装 |
| Memory Bank | 過去フレームの特徴を蓄積・管理 | ❌ 未実装 |
| Memory Attention | 現在フレームと過去メモリの関連付け | ❌ 未実装 |
| Object Pointer | オブジェクト追跡用のポインタ管理 | ❌ 未実装 |
| Temporal Propagation | 時間方向のマスク伝播 | ❌ 未実装 |

### 3.3 コード検証結果

以下のキーワードでソースコードを検索した結果、関連する実装が見つかりませんでした：

- `track`, `tracking` - 該当なし
- `video` - 該当なし
- `memory_bank` - 該当なし
- `propagat` - 該当なし
- `temporal`, `frame` - 該当なし（メモリ管理以外）
- `object_pointer` - 該当なし

### 3.4 現在の設計思想

本プロジェクトは**静止画像セグメンテーションに特化**した設計となっています：

```cpp
// sam3infer.hpp より
const int max_image_batch_ = 2;       // 同時処理画像数
const int max_prompt_batch_ = 4;      // デコードバッチサイズ
const int max_boxes_per_prompt_ = 20; // プロンプトあたりの最大ボックス数
```

フレーム間の時間的関係を保持する機構は実装されていません。

---

## 4. 高速化効果の定量的分析

### 4.1 推論速度

| 指標 | 値 | 条件 |
|-----|-----|------|
| **推論時間** | 約50ms/画像 | RTX 4090, 単一プロンプト |
| **スループット** | 約20 FPS | 連続推論時 |

### 4.2 ベンチマーク実装

```cpp
// main.cpp: speed_test()
nv::EventTimer timer;
timer.start();
for (int i = 0; i < 1000; i++)
    engine->forwards(inputs);
float ms = timer.stop();
printf("Inference 1000 images finished in %.2f ms, fps %f.\n",
       ms, 1000 / (ms / 1000));
```

### 4.3 高速化技術

#### TensorRT最適化
- **精度**: FP16/FP32混合精度
- **動的バッチ**: 可変バッチサイズに対応
- **TRTバージョン**: TensorRT 10.9.0.34 対応

#### GPU最適化
- **CUDAカーネル**: 前処理・後処理の専用CUDA実装
- **非同期実行**: `cudaMemcpyAsync`による並列データ転送
- **メモリ再利用**: 事前確保バッファによるアロケーション削減

```cpp
// 非同期メモリ転送の例
cudaMemcpyAsync(dst_0, src_0, sz_0 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
```

#### バッチ処理最適化
- **Vision Feature Gather**: 複数プロンプトで同一画像特徴を共有
- **特徴キャッシュ**: テキストトークン・Geometry特徴のキャッシュ

```cpp
// Vision特徴のGather（sam3infer.cpp）
void Sam3Infer::gather_vision_features(
    const std::vector<PromptMeta> &batch_prompts,
    int batch_size, void *stream) {
    // 複数プロンプトに対して同一画像の特徴を効率的にコピー
}
```

### 4.4 PyTorchとの速度比較（推定）

| 実装 | 推定速度 | 高速化率 |
|-----|---------|---------|
| PyTorch (FP32) | 200-300ms | 基準 |
| ONNX Runtime (CUDA) | 100-150ms | 2-3x |
| **TensorRT (本実装)** | **~50ms** | **4-6x** |

※ 推定値。実測にはPyTorch実装との直接比較が必要。

### 4.5 メモリ効率

```cpp
// 事前メモリ確保（sam3infer.cpp）
void Sam3Infer::allocate_memory_once() {
    // 固定サイズバッファの事前確保
    preprocessed_images_.gpu(max_image_batch_ * 3 * 1008 * 1008);
    fpn_feat_0_.gpu(max_image_batch_ * 256 * 288 * 288);
    // ... 他のバッファ
}
```

推定VRAM使用量: 約2-4GB（モデル + 処理バッファ）

---

## 5. コード品質分析

### 5.1 コード統計

| 項目 | 値 |
|-----|-----|
| 総C++/CUDAコード行数 | 約3,153行 |
| 主要ソースファイル数 | 15+ |
| Python バインディング | pybind11使用 |

### 5.2 アーキテクチャ評価

**強み:**
- TensorRT 8/10の両バージョン対応
- 動的バッチサイズ対応
- 包括的なPython API
- メモリ効率の良い設計

**改善点:**
- Streaming Tracking未対応
- ポイントプロンプト未対応
- エラーハンドリングの強化余地

---

## 6. まとめ

### 6.1 対応状況サマリー

| カテゴリ | 対応率 |
|---------|--------|
| 静止画像セグメンテーション | 100% |
| プロンプト種類 | 67% (2/3: テキスト、ボックス) |
| ビデオ/Streaming | 0% |

### 6.2 推奨事項

1. **Streaming Tracking実装**: ビデオセグメンテーションが必要な場合、Memory Encoder/Bank/Attentionの追加実装が必要
2. **ポイントプロンプト追加**: SAM互換性向上のため、ポイントプロンプト対応を検討
3. **ベンチマーク拡充**: PyTorch/ONNX Runtimeとの直接比較測定

---

## 7. 参考情報

### 7.1 動作環境

- OS: Ubuntu 24.04
- GPU: NVIDIA GeForce RTX 4090
- CUDA: 13.x / 11.8
- TensorRT: 10.9.0.34
- Docker: nvcr.io/nvidia/tensorrt:25.10-py3

### 7.2 関連リポジトリ

- ONNX エクスポート: [jamjamjon/usls](https://github.com/jamjamjon/usls)
- 事前学習済みモデル: [HuggingFace - tangliyang/onnx_model_store](https://huggingface.co/tangliyang/onnx_model_store)

---

*レポート作成日: 2025-12-26*
*分析対象: trt-sam3 (commit: de3c482)*
