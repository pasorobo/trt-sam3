#include "kernels/tracking_kernels.cuh"
#include <cfloat>

namespace cuda {

__global__ void compute_iou_matrix_kernel(
    const float* det_boxes,
    const float* track_boxes,
    float* iou_matrix,
    int num_dets,
    int num_tracks)
{
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (det_idx >= num_dets || track_idx >= num_tracks) {
        return;
    }

    // Get detection box (x1, y1, x2, y2)
    const float* det_box = det_boxes + det_idx * 4;
    float det_x1 = det_box[0];
    float det_y1 = det_box[1];
    float det_x2 = det_box[2];
    float det_y2 = det_box[3];

    // Get track box (x1, y1, x2, y2)
    const float* track_box = track_boxes + track_idx * 4;
    float track_x1 = track_box[0];
    float track_y1 = track_box[1];
    float track_x2 = track_box[2];
    float track_y2 = track_box[3];

    // Compute intersection
    float inter_x1 = fmaxf(det_x1, track_x1);
    float inter_y1 = fmaxf(det_y1, track_y1);
    float inter_x2 = fminf(det_x2, track_x2);
    float inter_y2 = fminf(det_y2, track_y2);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    // Compute areas
    float det_area = (det_x2 - det_x1) * (det_y2 - det_y1);
    float track_area = (track_x2 - track_x1) * (track_y2 - track_y1);
    float union_area = det_area + track_area - inter_area;

    // Compute IoU
    float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;

    // Store in matrix
    iou_matrix[det_idx * num_tracks + track_idx] = iou;
}

__global__ void find_best_matches_kernel(
    const float* iou_matrix,
    int* match_indices,
    float* match_scores,
    int num_dets,
    int num_tracks,
    float iou_threshold)
{
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (det_idx >= num_dets) {
        return;
    }

    float best_iou = iou_threshold;
    int best_track = -1;

    for (int t = 0; t < num_tracks; ++t) {
        float iou = iou_matrix[det_idx * num_tracks + t];
        if (iou > best_iou) {
            best_iou = iou;
            best_track = t;
        }
    }

    match_indices[det_idx] = best_track;
    match_scores[det_idx] = (best_track >= 0) ? best_iou : 0.0f;
}

__global__ void gather_memory_features_kernel(
    const float* src_memories,
    float* dst_memories,
    const int* frame_indices,
    int num_frames,
    int memory_size)
{
    int frame_idx = blockIdx.y;
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (frame_idx >= num_frames || elem_idx >= memory_size) {
        return;
    }

    int src_frame = frame_indices[frame_idx];
    dst_memories[frame_idx * memory_size + elem_idx] =
        src_memories[src_frame * memory_size + elem_idx];
}

__global__ void apply_mask_to_features_kernel(
    const float* features,
    const float* mask,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_elements = batch_size * channels * spatial_size;

    if (idx >= total_elements) {
        return;
    }

    int b = idx / (channels * spatial_size);
    int remaining = idx % (channels * spatial_size);
    int c = remaining / spatial_size;
    int spatial_idx = remaining % spatial_size;

    // Mask is [B, 1, H, W]
    float mask_val = mask[b * spatial_size + spatial_idx];
    output[idx] = features[idx] * mask_val;
}

__global__ void fuse_memory_features_kernel(
    const float* current_features,
    const float* memory_features,
    float* output_features,
    float alpha,
    int batch_size,
    int channels,
    int spatial_size,
    int num_memories)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;

    if (idx >= total_elements) {
        return;
    }

    // Average memory features
    float mem_sum = 0.0f;
    int c_spatial = idx % (channels * spatial_size);
    for (int m = 0; m < num_memories; ++m) {
        mem_sum += memory_features[m * channels * spatial_size + c_spatial];
    }
    float mem_avg = (num_memories > 0) ? (mem_sum / num_memories) : 0.0f;

    // Fuse with current features
    float current_val = current_features[idx];
    output_features[idx] = (1.0f - alpha) * current_val + alpha * mem_avg;
}

__global__ void compute_mask_iou_kernel(
    const unsigned char* mask1,
    const unsigned char* mask2,
    int* intersection,
    int* union_area,
    int height,
    int width)
{
    __shared__ int shared_inter[256];
    __shared__ int shared_union[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;

    int local_inter = 0;
    int local_union = 0;

    if (idx < total_pixels) {
        int m1 = (mask1[idx] > 0) ? 1 : 0;
        int m2 = (mask2[idx] > 0) ? 1 : 0;
        local_inter = m1 & m2;
        local_union = m1 | m2;
    }

    shared_inter[tid] = local_inter;
    shared_union[tid] = local_union;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_inter[tid] += shared_inter[tid + stride];
            shared_union[tid] += shared_union[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(intersection, shared_inter[0]);
        atomicAdd(union_area, shared_union[0]);
    }
}

__global__ void resize_mask_kernel(
    const float* src,
    float* dst,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) {
        return;
    }

    // Compute source coordinates (nearest neighbor)
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    int src_x = (int)(dst_x * scale_x);
    int src_y = (int)(dst_y * scale_y);

    src_x = min(src_x, src_w - 1);
    src_y = min(src_y, src_h - 1);

    dst[dst_y * dst_w + dst_x] = src[src_y * src_w + src_x];
}

// Host wrapper functions
void compute_iou_matrix(
    const float* det_boxes,
    const float* track_boxes,
    float* iou_matrix,
    int num_dets,
    int num_tracks,
    cudaStream_t stream)
{
    if (num_dets == 0 || num_tracks == 0) {
        return;
    }

    dim3 block(16, 16);
    dim3 grid((num_dets + block.x - 1) / block.x,
              (num_tracks + block.y - 1) / block.y);

    compute_iou_matrix_kernel<<<grid, block, 0, stream>>>(
        det_boxes, track_boxes, iou_matrix, num_dets, num_tracks);
}

void find_best_matches(
    const float* iou_matrix,
    int* match_indices,
    float* match_scores,
    int num_dets,
    int num_tracks,
    float iou_threshold,
    cudaStream_t stream)
{
    if (num_dets == 0) {
        return;
    }

    int block_size = 256;
    int grid_size = (num_dets + block_size - 1) / block_size;

    find_best_matches_kernel<<<grid_size, block_size, 0, stream>>>(
        iou_matrix, match_indices, match_scores, num_dets, num_tracks, iou_threshold);
}

void gather_memory_features(
    const float* src_memories,
    float* dst_memories,
    const int* frame_indices,
    int num_frames,
    int memory_size,
    cudaStream_t stream)
{
    if (num_frames == 0) {
        return;
    }

    int block_size = 256;
    int grid_x = (memory_size + block_size - 1) / block_size;

    dim3 grid(grid_x, num_frames);
    dim3 block(block_size);

    gather_memory_features_kernel<<<grid, block, 0, stream>>>(
        src_memories, dst_memories, frame_indices, num_frames, memory_size);
}

void fuse_memory_features(
    const float* current_features,
    const float* memory_features,
    float* output_features,
    float alpha,
    int batch_size,
    int channels,
    int spatial_size,
    int num_memories,
    cudaStream_t stream)
{
    int total_elements = batch_size * channels * spatial_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    fuse_memory_features_kernel<<<grid_size, block_size, 0, stream>>>(
        current_features, memory_features, output_features,
        alpha, batch_size, channels, spatial_size, num_memories);
}

void resize_mask(
    const float* src,
    float* dst,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    resize_mask_kernel<<<grid, block, 0, stream>>>(
        src, dst, src_h, src_w, dst_h, dst_w);
}

} // namespace cuda
