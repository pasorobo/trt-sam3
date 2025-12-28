"""
SAM3 Video Tracking Demo

This demo shows how to use the Sam3Tracker for video object tracking.
Objects are detected in the first frame and tracked throughout the video.
"""

import argparse
import cv2
import numpy as np
import os
import random
import time
from pathlib import Path
from tokenizers import Tokenizer
import trtsam3


# --- Configuration ---
ENGINE_DIR = "engine-models"
VISION_MODEL = f"{ENGINE_DIR}/vision-encoder.engine"
TEXT_MODEL = f"{ENGINE_DIR}/text-encoder.engine"
DECODER_MODEL = f"{ENGINE_DIR}/decoder.engine"
GEOMETRY_MODEL = f"{ENGINE_DIR}/geometry-encoder.engine"
MEMORY_ENCODER_MODEL = f"{ENGINE_DIR}/memory-encoder.engine"
MEMORY_ATTENTION_MODEL = f"{ENGINE_DIR}/memory-attention.engine"
TOKENIZER_PATH = f"{ENGINE_DIR}/tokenizer.json"
OUTPUT_DIR = "output"


def get_color_for_id(object_id: int) -> tuple:
    """Generate a consistent color for each object ID."""
    random.seed(object_id * 42)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_tracking_results(image: np.ndarray, results: list, show_trails: bool = True,
                          trails: dict = None) -> np.ndarray:
    """
    Visualize tracking results on the image.

    Args:
        image: Input image (BGR)
        results: List of TrackingResult objects
        show_trails: Whether to show object movement trails
        trails: Dictionary mapping object_id to list of past center positions

    Returns:
        Visualized image
    """
    vis = image.copy()

    for result in results:
        obj_id = result.object_id
        color = get_color_for_id(obj_id)

        # Get bounding box
        x1, y1, x2, y2 = [int(v) for v in result.bbox]

        # Clamp to image bounds
        h, w = vis.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw mask if available
        mask = result.mask
        if mask is not None and mask.size > 0:
            box_w, box_h = x2 - x1, y2 - y1
            if box_w > 0 and box_h > 0:
                # Resize mask to match box size
                if mask.shape[0] != box_h or mask.shape[1] != box_w:
                    mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

                # Create colored overlay
                roi = vis[y1:y2, x1:x2]
                colored_layer = np.zeros_like(roi)
                colored_layer[:] = color

                mask_indices = mask > 0
                if mask_indices.any():
                    roi[mask_indices] = cv2.addWeighted(
                        roi[mask_indices], 0.5,
                        colored_layer[mask_indices], 0.5, 0
                    )
                    vis[y1:y2, x1:x2] = roi

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label with object ID and confidence
        status = "NEW" if result.is_new else f"ID:{obj_id}"
        label = f"{status} {result.class_name}: {result.confidence:.2f}"
        text_y = max(y1 - 5, 20)
        cv2.putText(vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update and draw trails
        if show_trails and trails is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if obj_id not in trails:
                trails[obj_id] = []
            trails[obj_id].append((cx, cy))
            # Keep only last 30 positions
            trails[obj_id] = trails[obj_id][-30:]

            # Draw trail
            if len(trails[obj_id]) > 1:
                pts = np.array(trails[obj_id], dtype=np.int32)
                cv2.polylines(vis, [pts], False, color, 2)

    return vis


def init_tracker(gpu_id: int = 0, use_memory: bool = False) -> trtsam3.Sam3Tracker:
    """
    Initialize the SAM3 Tracker.

    Args:
        gpu_id: GPU device ID
        use_memory: Whether to use memory-based tracking (requires memory engines)

    Returns:
        Sam3Tracker instance
    """
    print(f"Initializing Sam3Tracker on GPU {gpu_id}...")

    config = trtsam3.Sam3TrackerConfig()
    config.vision_encoder_path = VISION_MODEL
    config.text_encoder_path = TEXT_MODEL
    config.geometry_encoder_path = GEOMETRY_MODEL
    config.decoder_path = DECODER_MODEL
    config.gpu_id = gpu_id

    # Memory engines (optional)
    if use_memory:
        if os.path.exists(MEMORY_ENCODER_MODEL):
            config.memory_encoder_path = MEMORY_ENCODER_MODEL
        if os.path.exists(MEMORY_ATTENTION_MODEL):
            config.memory_attention_path = MEMORY_ATTENTION_MODEL

    # Tracking parameters
    config.detection_threshold = 0.5
    config.tracking_threshold = 0.3
    config.match_iou_threshold = 0.5
    config.max_lost_frames = 30

    # Memory bank configuration
    config.memory_config.max_recent_frames = 6
    config.memory_config.max_prompted_frames = 2

    tracker = trtsam3.Sam3Tracker.create_instance(config)
    if tracker is None:
        raise RuntimeError("Failed to create Sam3Tracker instance.")

    print("  ✓ Tracker initialized")
    return tracker


def load_tokenizer() -> Tokenizer:
    """Load and configure the CLIP tokenizer."""
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_padding(length=32, pad_id=49407)
    tokenizer.enable_truncation(max_length=32)
    return tokenizer


def track_video(
    video_path: str,
    prompts: list,
    output_path: str = None,
    show_preview: bool = True,
    max_frames: int = None,
    use_memory: bool = False,
):
    """
    Track objects in a video based on text prompts.

    Args:
        video_path: Path to input video
        prompts: List of text prompts (e.g., ["person", "car"])
        output_path: Path to save output video (optional)
        show_preview: Whether to show live preview
        max_frames: Maximum number of frames to process (optional)
        use_memory: Whether to use memory-based tracking
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Prompts: {prompts}")

    # Initialize tracker
    tracker = init_tracker(use_memory=use_memory)
    tokenizer = load_tokenizer()

    # Create prompt units
    prompt_units = []
    for text in prompts:
        encoded = tokenizer.encode(text)
        prompt_units.append(trtsam3.Sam3PromptUnit(text))

    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Object trails for visualization
    trails = {}

    # Process video
    frame_count = 0
    total_time = 0
    initialized = False

    print("\nProcessing video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            start_time = time.time()

            if not initialized:
                # Initialize tracker with first frame
                success = tracker.initialize(frame, prompt_units, 0.5)
                if success:
                    initialized = True
                    print(f"  Frame 0: Tracker initialized with {len(tracker.get_tracked_object_ids())} objects")
                else:
                    print("  Warning: No objects detected in first frame. Continuing...")

                results = []
            else:
                # Track subsequent frames
                results = tracker.track_frame(frame)

            elapsed = time.time() - start_time
            total_time += elapsed

            # Visualize results
            vis_frame = draw_tracking_results(frame, results, show_trails=True, trails=trails)

            # Draw stats
            fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: N/A"
            obj_text = f"Objects: {len(results)}"
            cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_frame, obj_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Write to output
            if writer:
                writer.write(vis_frame)

            # Show preview
            if show_preview:
                cv2.imshow("SAM3 Tracking", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user.")
                    break
                elif key == ord('r'):
                    # Reset tracker
                    tracker.reset()
                    initialized = False
                    trails.clear()
                    print("  Tracker reset.")

            frame_count += 1

            # Progress update
            if frame_count % 100 == 0:
                avg_fps = frame_count / total_time if total_time > 0 else 0
                print(f"  Processed {frame_count}/{total_frames} frames (avg {avg_fps:.1f} FPS)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Summary
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessing complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    if output_path:
        print(f"  Output saved to: {output_path}")


def track_webcam(prompts: list, use_memory: bool = False):
    """
    Track objects from webcam feed.

    Args:
        prompts: List of text prompts
        use_memory: Whether to use memory-based tracking
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")

    tracker = init_tracker(use_memory=use_memory)
    tokenizer = load_tokenizer()

    # Create prompt units
    prompt_units = []
    for text in prompts:
        prompt_units.append(trtsam3.Sam3PromptUnit(text))

    trails = {}
    initialized = False

    print(f"\nWebcam tracking started. Prompts: {prompts}")
    print("Press 'q' to quit, 'r' to reset tracker, 'i' to re-initialize.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            start_time = time.time()

            if not initialized:
                success = tracker.initialize(frame, prompt_units, 0.5)
                initialized = success
                results = []
            else:
                results = tracker.track_frame(frame)

            elapsed = time.time() - start_time

            # Visualize
            vis_frame = draw_tracking_results(frame, results, show_trails=True, trails=trails)

            fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: N/A"
            cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Objects: {len(results)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("SAM3 Webcam Tracking", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                initialized = False
                trails.clear()
                print("Tracker reset.")
            elif key == ord('i'):
                initialized = False
                trails.clear()
                print("Re-initializing on next frame...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Tracking Demo")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument("--prompts", type=str, nargs="+", default=["person"],
                        help="Text prompts for tracking (e.g., --prompts person car)")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--use-memory", action="store_true",
                        help="Use memory-based tracking (requires memory engines)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.webcam:
        track_webcam(args.prompts, use_memory=args.use_memory)
    elif args.video:
        output_path = args.output or os.path.join(OUTPUT_DIR, "tracking_output.mp4")
        track_video(
            args.video,
            args.prompts,
            output_path=output_path,
            show_preview=not args.no_preview,
            max_frames=args.max_frames,
            use_memory=args.use_memory,
        )
    else:
        parser.error("Please specify --video or --webcam")


if __name__ == "__main__":
    main()
