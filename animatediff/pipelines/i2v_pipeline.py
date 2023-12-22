# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py
import inspect
import os.path as osp
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders import IPAdapterMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import (BaseOutput, deprecate, is_accelerate_available,
                             logging)
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from safetensors import safe_open
from tqdm import tqdm
from transformers import (CLIPImageProcessor, CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection)

from animatediff.models.resnet import InflatedConv3d
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.convert_from_ckpt import (convert_ldm_clip_checkpoint,
                                                 convert_ldm_unet_checkpoint,
                                                 convert_ldm_vae_checkpoint)
from animatediff.utils.convert_lora_safetensor_to_diffusers import \
    convert_lora_model_level
from animatediff.utils.util import prepare_mask_coef_by_statistics

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


DEFAULT_N_PROMPT = ('wrong white balance, dark, sketches,worst quality,'
                    'low quality, deformed, distorted, disfigured, bad eyes, '
                    'wrong lips,weird mouth, bad teeth, mutated hands and fingers, '
                    'bad anatomy,wrong anatomy, amputation, extra limb, '
                    'missing limb, floating,limbs, disconnected limbs, mutation, '
                    'ugly, disgusting, bad_pictures, negative_hand-neg')


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class I2VPipeline(DiffusionPipeline, IPAdapterMixin, TextualInversionLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        # memory_format: torch.memory_format,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.memory_format = memory_format
        self.use_ip_adapter = False

    @classmethod
    def build_pipeline(cls,
                       base_cfg,
                       base_model: str,
                       unet_path: str,
                       dreambooth_path: Optional[str] = None,
                       lora_path: Optional[str] = None,
                       lora_alpha: int = 0,
                       vae_path: Optional[str] = None,
                       ip_adapter_path: Optional[str] = None,
                       ip_adapter_scale: float = 0.0,
                       only_load_vae_decoder: bool = False,
                       only_load_vae_encoder: bool = False) -> 'I2VPipeline':
        """Method to build pipeline in a faster way~
        Args:
            base_cfg: The config to build model
            base_mode: The model id to initialize StableDiffusion
            unet_path: Path for i2v unet

            dreambooth_path: path for dreambooth model
            lora_path: path for lora model
            lora_alpha: value for lora scale

            only_load_vae_decoder: Only load VAE decoder from dreambooth / VAE ckpt
                and maitain encoder as original.

        """
        # build unet
        unet = UNet3DConditionModel.from_pretrained_2d(
            base_model, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                base_cfg.unet_additional_kwargs))

        old_weights = unet.conv_in.weight
        old_bias = unet.conv_in.bias
        new_conv1 = InflatedConv3d(
            9, old_weights.shape[0],
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            bias=True if old_bias is not None else False)
        param = torch.zeros((320,5,3,3),requires_grad=True)
        new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights,param),dim=1))
        if old_bias is not None:
            new_conv1.bias = old_bias
        unet.conv_in = new_conv1
        unet.config["in_channels"] = 9

        unet_ckpt = torch.load(unet_path, map_location='cpu')
        unet.load_state_dict(unet_ckpt, strict=False)
        # NOTE: only load temporal layers and condition module
        # for key, value in unet_ckpt.items():
        #     if 'motion' in key or 'conv_in' in key:
        #         unet.state_dict()[key].copy_(value)

        # load vae, tokenizer, text encoder
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        noise_scheduler = DDIMScheduler(**OmegaConf.to_container(base_cfg.noise_scheduler_kwargs))

        if dreambooth_path:

            print(" >>> Begin loading DreamBooth >>>")
            base_model_state_dict = {}
            with safe_open(dreambooth_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)

            # load unet
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, unet.config)

            old_value = converted_unet_checkpoint['conv_in.weight']
            new_param = unet_ckpt['conv_in.weight'][:,4:,:,:].clone().cpu()
            new_value = torch.nn.Parameter(torch.cat((old_value, new_param), dim=1))
            converted_unet_checkpoint['conv_in.weight'] = new_value
            unet.load_state_dict(converted_unet_checkpoint, strict=False)

            # load vae
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                base_model_state_dict, vae.config,
                only_decoder=only_load_vae_decoder,
                only_encoder=only_load_vae_encoder,)
            need_strict = not (only_load_vae_decoder or only_load_vae_encoder)
            vae.load_state_dict(converted_vae_checkpoint, strict=need_strict)
            print('Prefix in loaded VAE checkpoint: ')
            print(set([k.split('.')[0] for k in converted_vae_checkpoint.keys()]))

            # load text encoder
            text_encoder_checkpoint = convert_ldm_clip_checkpoint(base_model_state_dict)
            if text_encoder_checkpoint:
                text_encoder.load_state_dict(text_encoder_checkpoint)

            print(" <<< Loaded DreamBooth        <<<")

        if vae_path:
            print(' >>> Begin loading VAE >>>')
            vae_state_dict = {}
            if vae_path.endswith('safetensors'):
                with safe_open(vae_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        vae_state_dict[key] = f.get_tensor(key)
            elif vae_path.endswith('ckpt') or vae_path.endswith('pt'):
                vae_state_dict = torch.load(vae_path, map_location='cpu')
            if 'state_dict' in vae_state_dict:
                vae_state_dict = vae_state_dict['state_dict']

            vae_state_dict = {f'first_stage_model.{k}': v for k, v in vae_state_dict.items()}

            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                vae_state_dict, vae.config,
                only_decoder=only_load_vae_decoder,
                only_encoder=only_load_vae_encoder,)
            print('Prefix in loaded VAE checkpoint: ')
            print(set([k.split('.')[0] for k in converted_vae_checkpoint.keys()]))
            need_strict = not (only_load_vae_decoder or only_load_vae_encoder)
            vae.load_state_dict(converted_vae_checkpoint, strict=need_strict)
            print(" <<< Loaded VAE        <<<")

        if lora_path:

            print(" >>> Begin loading LoRA >>>")

            lora_dict = {}
            with safe_open(lora_path, framework='pt', device='cpu') as file:
                for k in file.keys():
                    lora_dict[k] = file.get_tensor(k)
            unet, text_encoder = convert_lora_model_level(
                lora_dict, unet, text_encoder, alpha=lora_alpha)

            print(" <<< Loaded LoRA        <<<")

        # move model to device
        device = torch.device('cuda')
        unet_dtype = torch.float16
        tenc_dtype = torch.float16
        vae_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        unet = unet.to(device=device, dtype=unet_dtype)
        text_encoder = text_encoder.to(device=device, dtype=tenc_dtype)
        vae = vae.to(device=device, dtype=vae_dtype)
        print(f'Set Unet to {unet_dtype}')
        print(f'Set text encoder to {tenc_dtype}')
        print(f'Set vae to {vae_dtype}')

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        pipeline = cls(unet=unet,
                       vae=vae,
                       tokenizer=tokenizer,
                       text_encoder=text_encoder,
                       scheduler=noise_scheduler)

        # ip_adapter_path = 'h94/IP-Adapter'
        if ip_adapter_path and ip_adapter_scale > 0:
            ip_adapter_name = 'ip-adapter_sd15.bin'
            # only online repo need subfolder
            if not osp.isdir(ip_adapter_path):
                subfolder = 'models'
            else:
                subfolder = ''
            pipeline.load_ip_adapter(ip_adapter_path, subfolder, ip_adapter_name)
            pipeline.set_ip_adapter_scale(ip_adapter_scale)
            pipeline.use_ip_adapter = True
            print(f'Load IP-Adapter, scale: {ip_adapter_scale}')

        # text_inversion_path = './models/TextualInversion/easynegative.safetensors'
        # if text_inversion_path:
        #     pipeline.load_textual_inversion(text_inversion_path, 'easynegative')

        return pipeline

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, add_noise_time_step, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        return latents

    def encode_image(self, image, device, num_images_per_prompt):
        """Encode image for ip-adapter. Copied from
        https://github.com/huggingface/diffusers/blob/f9487783228cd500a21555da3346db40e8f05992/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L492-L514  # noqa
        """
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    @torch.no_grad()
    def __call__(
        self,
        image: np.ndarray,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        global_inf_num: int = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        cond_frame: int = 0,
        mask_sim_template_idx: int = 0,
        ip_adapter_scale: float = 0,
        strength: float = 1,
        progress_fn=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        assert strength > 0 and strength <= 1, (
            f'"strength" for img2vid must in (0, 1]. But receive {strength}.')

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size

        if negative_prompt is None:
            negative_prompt = DEFAULT_N_PROMPT
        negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        #timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size)

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            latent_timestep,
            batch_size * num_videos_per_prompt,
            4,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)

        raw_image = image.copy()
        image = torch.from_numpy(image)[None, ...].permute(0, 3, 1, 2)
        image = image / 255  # [0, 1]
        image = image * 2 - 1   # [-1, 1]
        image = image.to(device=device, dtype=self.vae.dtype)

        if isinstance(generator, list):
            image_latent = [
                self.vae.encode(image[k : k + 1]).latent_dist.sample(generator[k]) for k in range(batch_size)
            ]
            image_latent = torch.cat(image_latent, dim=0)
        else:
            image_latent = self.vae.encode(image).latent_dist.sample(generator)

        image_latent = image_latent.to(device=device, dtype=self.unet.dtype)
        image_latent = torch.nn.functional.interpolate(image_latent, size=[shape[-2], shape[-1]])
        image_latent_padding = image_latent.clone() * 0.18215
        mask = torch.zeros((shape[0], 1, shape[2], shape[3], shape[4])).to(device=device, dtype=self.unet.dtype)

        # prepare mask
        mask_coef = prepare_mask_coef_by_statistics(video_length, cond_frame, mask_sim_template_idx)

        masked_image = torch.zeros(shape[0], 4, shape[2], shape[3], shape[4]).to(device=device, dtype=self.unet.dtype)
        for f in range(video_length):
            mask[:,:,f,:,:]         = mask_coef[f]
            masked_image[:,:,f,:,:] = image_latent_padding.clone()

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image = torch.cat([masked_image] * 2) if do_classifier_free_guidance else masked_image
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # prepare for ip-adapter
        if self.use_ip_adapter:
            image_embeds, neg_image_embeds = self.encode_image(raw_image, device, num_videos_per_prompt)
            image_embeds = torch.cat([neg_image_embeds, image_embeds])
            image_embeds = image_embeds.to(device=device, dtype=self.unet.dtype)

            self.set_ip_adapter_scale(ip_adapter_scale)
            print(f'Set IP-Adapter Scale as {ip_adapter_scale}')

        else:

            image_embeds = None

        # prepare for latents if strength < 1, add convert gaussian latent to masked_img and add noise
        if strength < 1:
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(masked_image[0], noise, timesteps[0])
            print(latents.shape)

        if progress_fn is None:
            progress_bar = tqdm(timesteps)
            terminal_pbar = None
        else:
            progress_bar = progress_fn.tqdm(timesteps)
            terminal_pbar = tqdm(total=len(timesteps))

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(progress_bar):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                mask,
                masked_image,
                t,
                encoder_hidden_states=text_embeddings,
                image_embeds=image_embeds
            )['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            if terminal_pbar is not None:
                terminal_pbar.update(1)

        # Post-processing
        video = self.decode_latents(latents.to(device, dtype=self.vae.dtype))

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)