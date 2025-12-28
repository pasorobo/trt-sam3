#ifndef TRACKING_KERNELS_CUH__
#define TRACKING_KERNELS_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {

/**
 * @brief Compute IoU between detection boxes and tracked object boxes
 * @param det_boxes Detection boxes [N, 4] in (x1, y1, x2, y2) format
 * @param track_boxes Tracked object boxes [M, 4] in (x1, y1, x2, y2) format
 * @param iou_matrix Output IoU matrix [N, M]
 * @param num_dets Number of detections
 * @param num_tracks Number of tracked objects
 */
__global__ void compute_iou_matrix_kernel(
    const float* det_boxes,
    const float* track_boxes,
    float* iou_matrix,
    int num_dets,
    int num_tracks);

/**
 * @brief Find best matching track for each detection based on IoU
 * @param iou_matrix IoU matrix [N, M]
 * @param match_indices Output match indices for each detection (-1 if no match)
 * @param match_scores Output best IoU score for each detection
 * @param num_dets Number of detections
 * @param num_tracks Number of tracked objects
 * @param iou_threshold Minimum IoU for valid match
 */
__global__ void find_best_matches_kernel(
    const float* iou_matrix,
    int* match_indices,
    float* match_scores,
    int num_dets,
    int num_tracks,
    float iou_threshold);

/**
 * @brief Gather memory features for attention
 * @param src_memories Source memory features [num_frames, C, H, W]
 * @param dst_memories Destination gathered memories [max_memories, C, H, W]
 * @param frame_indices Indices of frames to gather
 * @param num_frames Number of frames to gather
 * @param memory_size Size of each memory (C * H * W)
 */
__global__ void gather_memory_features_kernel(
    const float* src_memories,
    float* dst_memories,
    const int* frame_indices,
    int num_frames,
    int memory_size);

/**
 * @brief Apply mask to memory features (element-wise multiply)
 * @param features Input features [N, C, H, W]
 * @param mask Binary mask [N, 1, H, W]
 * @param output Output masked features [N, C, H, W]
 * @param batch_size Number of samples
 * @param channels Number of channels
 * @param height Feature height
 * @param width Feature width
 */
__global__ void apply_mask_to_features_kernel(
    const float* features,
    const float* mask,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width);

/**
 * @brief Fuse current frame features with memory features
 * @param current_features Current frame features [B, C, H, W]
 * @param memory_features Memory features [num_memories, C, H, W]
 * @param output_features Fused output features [B, C, H, W]
 * @param alpha Blending weight for memory features (0-1)
 * @param batch_size Batch size
 * @param channels Number of channels
 * @param spatial_size H * W
 * @param num_memories Number of memory frames
 */
__global__ void fuse_memory_features_kernel(
    const float* current_features,
    const float* memory_features,
    float* output_features,
    float alpha,
    int batch_size,
    int channels,
    int spatial_size,
    int num_memories);

/**
 * @brief Compute mask IoU between two masks on GPU
 * @param mask1 First mask [H, W]
 * @param mask2 Second mask [H, W]
 * @param intersection Output intersection area
 * @param union_area Output union area
 * @param height Mask height
 * @param width Mask width
 */
__global__ void compute_mask_iou_kernel(
    const unsigned char* mask1,
    const unsigned char* mask2,
    int* intersection,
    int* union_area,
    int height,
    int width);

/**
 * @brief Resize mask using nearest neighbor interpolation
 * @param src Source mask
 * @param dst Destination mask
 * @param src_h Source height
 * @param src_w Source width
 * @param dst_h Destination height
 * @param dst_w Destination width
 */
__global__ void resize_mask_kernel(
    const float* src,
    float* dst,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w);

// Host wrapper functions
void compute_iou_matrix(
    const float* det_boxes,
    const float* track_boxes,
    float* iou_matrix,
    int num_dets,
    int num_tracks,
    cudaStream_t stream = nullptr);

void find_best_matches(
    const float* iou_matrix,
    int* match_indices,
    float* match_scores,
    int num_dets,
    int num_tracks,
    float iou_threshold,
    cudaStream_t stream = nullptr);

void gather_memory_features(
    const float* src_memories,
    float* dst_memories,
    const int* frame_indices,
    int num_frames,
    int memory_size,
    cudaStream_t stream = nullptr);

void fuse_memory_features(
    const float* current_features,
    const float* memory_features,
    float* output_features,
    float alpha,
    int batch_size,
    int channels,
    int spatial_size,
    int num_memories,
    cudaStream_t stream = nullptr);

void resize_mask(
    const float* src,
    float* dst,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    cudaStream_t stream = nullptr);

} // namespace cuda

#endif // TRACKING_KERNELS_CUH__
