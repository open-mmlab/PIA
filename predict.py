# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import os.path as osp
from glob import glob
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from cog import BasePredictor, Input, Path

from animatediff.pipelines import I2VPipeline
from animatediff.utils.util import save_videos_grid


N_PROMPT = (
    "wrong white balance, dark, sketches,worst quality,low quality, "
    "deformed, distorted, disfigured, bad eyes, wrong lips, "
    "weird mouth, bad teeth, mutated hands and fingers, bad anatomy,"
    "wrong anatomy, amputation, extra limb, missing limb, "
    "floating,limbs, disconnected limbs, mutation, ugly, disgusting, "
    "bad_pictures, negative_hand-neg"
)


BASE_CONFIG = "example/config/base.yaml"
STYLE_CONFIG_LIST = {
    "realistic": "example/replicate/1-realistic.yaml",
    "3d_cartoon": "example/replicate/3-3d.yaml",
}


PIA_PATH = "models/PIA"
VAE_PATH = "models/VAE"
DreamBooth_LoRA_PATH = "models/DreamBooth_LoRA"
STABLE_DIFFUSION_PATH = "models/StableDiffusion"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.ip_adapter_dir = (
            "models/IP_Adapter/h94/IP-Adapter/models"  # cached h94/IP-Adapter
        )

        self.inference_config = OmegaConf.load("example/config/base.yaml")
        self.stable_diffusion_dir = self.inference_config.pretrained_model_path
        self.pia_path = self.inference_config.generate.model_path
        self.style_configs = {
            k: OmegaConf.load(v) for k, v in STYLE_CONFIG_LIST.items()
        }
        self.pipeline_dict = self.load_model_list()

    def load_model_list(self):
        pipeline_dict = dict()
        for style, cfg in self.style_configs.items():
            print(f"Loading {style}")
            dreambooth_path = cfg.get("dreambooth", "none")
            if dreambooth_path and dreambooth_path.upper() != "NONE":
                dreambooth_path = osp.join(DreamBooth_LoRA_PATH, dreambooth_path)
            lora_path = cfg.get("lora", None)
            if lora_path is not None:
                lora_path = osp.join(DreamBooth_LoRA_PATH, lora_path)
            lora_alpha = cfg.get("lora_alpha", 0.0)
            vae_path = cfg.get("vae", None)
            if vae_path is not None:
                vae_path = osp.join(VAE_PATH, vae_path)

            pipeline_dict[style] = I2VPipeline.build_pipeline(
                self.inference_config,
                STABLE_DIFFUSION_PATH,
                unet_path=osp.join(PIA_PATH, "pia.ckpt"),
                dreambooth_path=dreambooth_path,
                lora_path=lora_path,
                lora_alpha=lora_alpha,
                vae_path=vae_path,
                ip_adapter_path=self.ip_adapter_dir,
                ip_adapter_scale=0.1,
            )
        return pipeline_dict

    def predict(
        self,
        prompt: str = Input(description="Input prompt."),
        image: Path = Input(description="Input image"),
        negative_prompt: str = Input(
            description="Things do not show in the output.", default=N_PROMPT
        ),
        style: str = Input(
            description="Choose a style",
            choices=["3d_cartoon", "realistic"],
            default="3d_cartoon",
        ),
        max_size: int = Input(
            description="Max size (The long edge of the input image will be resized to this value, "
            "larger value means slower inference speed)",
            default=512,
            choices=[512, 576, 640, 704, 768, 832, 896, 960, 1024],
        ),
        motion_scale: int = Input(
            description="Larger value means larger motion but less identity consistency.",
            ge=1,
            le=3,
            default=1,
        ),
        sampling_steps: int = Input(
            description="Number of denoising steps", ge=10, le=100, default=25
        ),
        animation_length: int = Input(
            description="Length of the output", ge=8, le=24, default=16
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=1.0,
            le=20.0,
            default=7.5,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            torch.seed()
            seed = torch.initial_seed()
        else:
            torch.manual_seed(seed)
        print(f"Using seed: {seed}")

        pipeline = self.pipeline_dict[style]

        init_img, h, w = preprocess_img(str(image), max_size)

        sample = pipeline(
            image=init_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=sampling_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=animation_length,
            mask_sim_template_idx=motion_scale,
            ip_adapter_scale=ip_adapter_scale,
        ).videos

        out_path = "/tmp/out.mp4"
        save_videos_grid(sample, out_path)
        return Path(out_path)


def preprocess_img(img_np, max_size: int = 512):
    ori_image = Image.open(img_np).convert("RGB")

    width, height = ori_image.size

    long_edge = max(width, height)
    if long_edge > max_size:
        scale_factor = max_size / long_edge
    else:
        scale_factor = 1
    width = int(width * scale_factor)
    height = int(height * scale_factor)
    ori_image = ori_image.resize((width, height))

    if (width % 8 != 0) or (height % 8 != 0):
        in_width = (width // 8) * 8
        in_height = (height // 8) * 8
    else:
        in_width = width
        in_height = height

    in_image = ori_image.resize((in_width, in_height))
    in_image_np = np.array(in_image)
    return in_image_np, in_height, in_width
