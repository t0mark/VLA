"""
UniVLA inference wrapper for ROS deployment.

Usage:
    model = UniVLAInference(
        model_path="/path/to/UNIVLA_LIBERO_IMG_BS192_8K",
        vision_hub="/path/to/Emu3-VisionVQ",
        fast_path="/path/to/fast",
        norm_stats_path="/path/to/norm_stats.json",
    )
    actions = model.predict(rgb_np, "pick up the red cube")
    # actions: np.ndarray (10, 7)  -- denormalized
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor, LogitsProcessor

# Make emu3 importable from this package
sys.path.insert(0, os.path.dirname(__file__))
from emu3.mllm import Emu3Tokenizer, Emu3MoE
from emu3.mllm.processing_emu3 import Emu3Processor


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


class UniVLAInference:
    def __init__(
        self,
        model_path: str,
        vision_hub: str,
        fast_path: str,
        norm_stats_path: str = None,
        device: str = "cuda:0",
        action_predict_frame: int = 10,
        action_dim: int = 7,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_predict_frame = action_predict_frame
        self.action_dim = action_dim

        print(f"[UniVLA] Loading model from {model_path}")
        self.model = Emu3MoE.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(self.device).eval()

        print(f"[UniVLA] Loading tokenizer / processor")
        self.tokenizer = Emu3Tokenizer.from_pretrained(
            model_path,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(vision_hub, trust_remote_code=True)
        self.image_processor.min_pixels = 80 * 80
        self.image_tokenizer = AutoModel.from_pretrained(
            vision_hub, trust_remote_code=True
        ).to(self.device).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        print(f"[UniVLA] Loading FAST action tokenizer from {fast_path}")
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

        eoa_token_id = 151845
        self.generation_config = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=eoa_token_id,
            do_sample=False,
        )
        last_token_id = self.tokenizer.pad_token_id - 1
        allowed_token_ids = (
            list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1))
            + [eoa_token_id]
        )
        self.action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
        self.last_token_id = last_token_id

        self.norm_stats = None
        if norm_stats_path and os.path.exists(norm_stats_path):
            with open(norm_stats_path) as f:
                data = json.load(f)
            # Support both {"norm_stats": {"libero": ...}} and {"mean": ..., "std": ...}
            stats = data.get("norm_stats", data)
            if isinstance(stats, dict) and not ("mean" in stats):
                # Nested by dataset name — take the first entry
                stats = next(iter(stats.values()))
            self.norm_stats = {
                "mean": np.array(stats["mean"]),
                "std": np.array(stats["std"]),
            }
            print(f"[UniVLA] Loaded norm stats from {norm_stats_path}")

        print("[UniVLA] Ready.")

    @torch.no_grad()
    def encode_image(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Encode a raw RGB image (H, W, 3) uint8 into VQ token codes.

        Returns: torch.Tensor of shape (1, H_tokens, W_tokens)
        """
        pil = Image.fromarray(rgb.astype(np.uint8))
        tokens = self.processor.tokenize_image([pil])  # List[Tensor(H_tok, W_tok)]
        return tokens[0].unsqueeze(0)  # (1, H_tok, W_tok)

    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        instruction: str,
        gripper_rgb: np.ndarray = None,
    ) -> np.ndarray:
        """
        Run one inference step.

        Args:
            rgb:           np.ndarray (H, W, 3) uint8  — main camera
            instruction:   str                          — task description
            gripper_rgb:   np.ndarray (H, W, 3) uint8  — optional gripper cam

        Returns:
            actions: np.ndarray (action_predict_frame, action_dim)
                     in robot's original units (denormalized if norm_stats provided)
        """
        video_code = self.encode_image(rgb)  # (1, H_tok, W_tok)

        gripper_code = None
        if gripper_rgb is not None:
            gripper_code = self.encode_image(gripper_rgb)  # (1, H_tok, W_tok)

        pos_inputs = self.processor.video_process(
            text=instruction,
            video_tokens=video_code,
            gripper_tokens=gripper_code,
            context_frames=1,
            frames=1,
            mode="VLA",
            padding="longest",
            return_tensors="pt",
        )

        outputs = self.model.generate(
            pos_inputs.input_ids.to(self.device),
            self.generation_config,
            max_new_tokens=50,
            logits_processor=[self.action_id_processor],
            attention_mask=pos_inputs.attention_mask.to(self.device),
        )
        # Strip input tokens and EOA
        outputs = outputs[:, pos_inputs.input_ids.shape[-1]:-1]

        last_token_id_tensor = torch.tensor(
            self.last_token_id, dtype=outputs.dtype, device=outputs.device
        )
        indices = last_token_id_tensor - outputs

        action_np = self.action_tokenizer.decode(
            indices,
            time_horizon=self.action_predict_frame,
            action_dim=self.action_dim,
        )  # (1, action_predict_frame, action_dim)
        action = action_np[0]  # (action_predict_frame, action_dim)

        if self.norm_stats is not None:
            action = action * self.norm_stats["std"] + self.norm_stats["mean"]

        return action  # np.ndarray (action_predict_frame, action_dim)
