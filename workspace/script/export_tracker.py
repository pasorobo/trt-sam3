"""
SAM3 Tracker ONNX Export Script

Exports Memory Encoder and Memory Attention modules for streaming tracking.
These modules are inherited from SAM2's memory mechanism.
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoderWrapper(nn.Module):
    """Memory Encoder: Encodes predicted masks and visual features into memory representations.

    Architecture (inherited from SAM2):
    - Mask downsampling via convolutions
    - Feature fusion with FPN features
    - Output: Spatial memory tensor [B, 256, 72, 72]

    Note: This is a placeholder implementation. The actual module should be loaded
    from the SAM3/SAM2 model when available.
    """

    def __init__(self, hidden_dim: int = 256, memory_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        # Mask downsampling layers (288x288 -> 72x72)
        self.mask_downsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4, padding=0),  # 288 -> 72
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, memory_dim, kernel_size=3, stride=1, padding=1),
        )

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim + memory_dim, memory_dim, kernel_size=1),
            nn.GroupNorm(16, memory_dim),
            nn.GELU(),
            nn.Conv2d(memory_dim, memory_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, memory_dim),
        )

        # Object pointer projection (from decoder tokens)
        self.pointer_proj = nn.Linear(hidden_dim, memory_dim)

    def forward(
        self,
        pred_mask: torch.Tensor,  # [B, 1, 288, 288] predicted mask
        fpn_feat_2: torch.Tensor,  # [B, 256, 72, 72] vision features
        decoder_tokens: Optional[torch.Tensor] = None,  # [B, num_queries, hidden_dim]
    ):
        """
        Encode the predicted mask and visual features into memory representation.

        Args:
            pred_mask: Predicted segmentation mask [B, 1, 288, 288]
            fpn_feat_2: FPN features at level 2 [B, 256, 72, 72]
            decoder_tokens: Optional decoder output tokens for object pointer [B, Q, D]

        Returns:
            memory: Spatial memory tensor [B, 256, 72, 72]
            object_pointer: Object pointer vector [B, 256] (if decoder_tokens provided)
        """
        batch_size = pred_mask.shape[0]

        # Downsample mask
        mask_feat = self.mask_downsample(pred_mask)  # [B, 256, 72, 72]

        # Fuse with vision features
        fused = torch.cat([fpn_feat_2, mask_feat], dim=1)  # [B, 512, 72, 72]
        memory = self.fusion(fused)  # [B, 256, 72, 72]

        # Extract object pointer from decoder tokens
        if decoder_tokens is not None:
            # Use the first token (CLS-like) as the object representation
            object_pointer = self.pointer_proj(decoder_tokens[:, 0, :])  # [B, 256]
            return memory, object_pointer

        return memory


class MemoryAttentionWrapper(nn.Module):
    """Memory Attention: Conditions current frame features with past memories.

    Architecture (inherited from SAM2):
    - 4-layer transformer with self-attention and cross-attention
    - Self-attention over memories + current features
    - Cross-attention from current features to memories
    - Output: Memory-conditioned features

    Note: This is a placeholder implementation. The actual module should be loaded
    from the SAM3/SAM2 model when available.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_memories: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_memories = max_memories

        # Pre-norm layer
        self.norm_current = nn.LayerNorm(hidden_dim)
        self.norm_memory = nn.LayerNorm(hidden_dim)

        # Attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Layer norms for each sublayer
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Object pointer attention
        self.pointer_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.pointer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        current_features: torch.Tensor,  # [B, H*W, D] current frame features
        memories: torch.Tensor,  # [num_memories, H*W, D] past frame memories
        memory_mask: torch.Tensor,  # [num_memories] validity mask
        object_pointers: Optional[torch.Tensor] = None,  # [num_memories, max_obj, D]
        pointer_mask: Optional[torch.Tensor] = None,  # [num_memories, max_obj]
    ):
        """
        Condition current frame features with past memories.

        Args:
            current_features: Current frame features [B, H*W, D]
            memories: Past frame memory tensors [num_memories, H*W, D]
            memory_mask: Validity mask for memories [num_memories]
            object_pointers: Object pointer vectors [num_memories, max_obj, D]
            pointer_mask: Validity mask for pointers [num_memories, max_obj]

        Returns:
            conditioned_features: Memory-conditioned features [B, H*W, D]
        """
        batch_size = current_features.shape[0]
        seq_len = current_features.shape[1]
        num_memories = memories.shape[0]

        # Normalize inputs
        current = self.norm_current(current_features)
        mem = self.norm_memory(memories)

        # Apply object pointer attention if available
        if object_pointers is not None and pointer_mask is not None:
            # Flatten pointers: [num_memories * max_obj, D]
            flat_pointers = object_pointers.reshape(-1, self.hidden_dim)
            flat_mask = pointer_mask.reshape(-1)

            # Filter valid pointers
            valid_pointers = flat_pointers[flat_mask]
            if valid_pointers.shape[0] > 0:
                # Add pointer information to current features
                pointer_ctx, _ = self.pointer_attn(
                    current,
                    valid_pointers.unsqueeze(0).expand(batch_size, -1, -1),
                    valid_pointers.unsqueeze(0).expand(batch_size, -1, -1),
                )
                current = self.pointer_norm(current + pointer_ctx)

        # Create key-value from memories (expand for batch)
        # Shape: [B, num_memories * H*W, D]
        mem_kv = mem.unsqueeze(0).expand(batch_size, -1, -1, -1)
        mem_kv = mem_kv.reshape(batch_size, num_memories * seq_len, self.hidden_dim)

        # Create attention mask
        # Shape: [B, seq_len, num_memories * seq_len]
        attn_mask = memory_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_memories]
        attn_mask = attn_mask.expand(batch_size, seq_len, num_memories)
        attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, -1, seq_len)
        attn_mask = attn_mask.reshape(batch_size, seq_len, num_memories * seq_len)
        attn_mask = ~attn_mask  # Invert for attention mask (True = ignore)

        # Apply attention layers
        x = current
        for i in range(self.num_layers):
            # Self-attention
            x_norm = self.norm1[i](x)
            self_attn_out, _ = self.self_attn_layers[i](x_norm, x_norm, x_norm)
            x = x + self_attn_out

            # Cross-attention to memories
            x_norm = self.norm2[i](x)
            cross_attn_out, _ = self.cross_attn_layers[i](
                x_norm, mem_kv, mem_kv,
                attn_mask=attn_mask if not attn_mask.all() else None
            )
            x = x + cross_attn_out

            # FFN
            x_norm = self.norm3[i](x)
            x = x + self.ffn_layers[i](x_norm)

        return x


def export_memory_encoder(output_dir: Path, device: str = "cuda"):
    """Export Memory Encoder to ONNX format."""
    print("Exporting Memory Encoder...")

    wrapper = MemoryEncoderWrapper(hidden_dim=256, memory_dim=256).to(device).eval()

    # Dummy inputs
    pred_mask = torch.randn(1, 1, 288, 288, device=device)
    fpn_feat_2 = torch.randn(1, 256, 72, 72, device=device)
    decoder_tokens = torch.randn(1, 100, 256, device=device)

    torch.onnx.export(
        wrapper,
        (pred_mask, fpn_feat_2, decoder_tokens),
        str(output_dir / "memory-encoder.onnx"),
        input_names=["pred_mask", "fpn_feat_2", "decoder_tokens"],
        output_names=["memory", "object_pointer"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "pred_mask": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "decoder_tokens": {0: "batch", 1: "num_queries"},
            "memory": {0: "batch"},
            "object_pointer": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'memory-encoder.onnx'}")


def export_memory_attention(output_dir: Path, device: str = "cuda"):
    """Export Memory Attention to ONNX format."""
    print("Exporting Memory Attention...")

    wrapper = MemoryAttentionWrapper(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        max_memories=8,
    ).to(device).eval()

    # Dummy inputs
    seq_len = 72 * 72  # H * W
    current_features = torch.randn(1, seq_len, 256, device=device)
    memories = torch.randn(4, seq_len, 256, device=device)  # 4 memory frames
    memory_mask = torch.ones(4, dtype=torch.bool, device=device)
    object_pointers = torch.randn(4, 10, 256, device=device)  # 4 frames, max 10 objects
    pointer_mask = torch.zeros(4, 10, dtype=torch.bool, device=device)
    pointer_mask[:, :3] = True  # First 3 objects valid

    torch.onnx.export(
        wrapper,
        (current_features, memories, memory_mask, object_pointers, pointer_mask),
        str(output_dir / "memory-attention.onnx"),
        input_names=[
            "current_features",
            "memories",
            "memory_mask",
            "object_pointers",
            "pointer_mask",
        ],
        output_names=["conditioned_features"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "current_features": {0: "batch", 1: "seq_len"},
            "memories": {0: "num_memories", 1: "seq_len"},
            "memory_mask": {0: "num_memories"},
            "object_pointers": {0: "num_memories", 1: "max_objects"},
            "pointer_mask": {0: "num_memories", 1: "max_objects"},
            "conditioned_features": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'memory-attention.onnx'}")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 Tracker modules to ONNX format"
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["memory-encoder", "memory-attention"],
        help="Module to export",
    )
    parser.add_argument("--all", action="store_true", help="Export all modules")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx-models",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Please specify --module or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = (
        ["memory-encoder", "memory-attention"]
        if args.all
        else [args.module]
    )

    with torch.no_grad():
        for m in modules:
            if m == "memory-encoder":
                export_memory_encoder(output_dir, args.device)
            elif m == "memory-attention":
                export_memory_attention(output_dir, args.device)

    print(f"\n✓ Export complete! Models saved to: {output_dir}")
    print("\nNote: These are placeholder implementations.")
    print("For production use, load the actual SAM3/SAM2 memory modules.")


if __name__ == "__main__":
    main()
