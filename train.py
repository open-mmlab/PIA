# largely borrowed from https://github.com/guoyww/AnimateDiff/blob/main/train.py
import argparse
import datetime
import inspect
import logging
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import wandb
from animatediff.data.dataset import WebVid10M
from animatediff.models.resnet import InflatedConv3d
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.validation_pipeline import ValidationPipeline
from animatediff.utils.util import prepare_mask_coef_by_score, save_videos_grid, zero_rank_print
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available


def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
        )

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def main(
    image_finetune: bool,
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs=None,
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    trainable_modules: Tuple[str] = (None,),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 32,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    statistic: list = [1, 40],
    global_seed: int = 42,
    is_debug: bool = False,
    mask_frame: list = [0],
    pretrained_motion_module_path: str = "",
    pretrained_sd_path: str = "",
    mask_sim_range: list = [0.2, 1.0],
    cond_prob: float = 0.2,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filemode="a",
        filename="train.log",
    )

    if is_main_process and (not is_debug) and use_wandb:
        # run = wandb.init(project="image2video", name=folder_name, config=config)
        wandb.init(project="image2video", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path:
            zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = (
            unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        )

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # assert len(u) == 0

    # Add additional five channels for conditional module in PIA
    old_weights = unet.conv_in.weight
    old_bias = unet.conv_in.bias
    new_conv1 = InflatedConv3d(
        9,
        old_weights.shape[0],
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=True if old_bias is not None else False,
    )
    param = torch.zeros((320, 5, 3, 3), requires_grad=True)
    new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights, param), dim=1))
    if old_bias is not None:
        new_conv1.bias = old_bias
    unet.conv_in = new_conv1
    unet.config["in_channels"] = 9

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                logging.info(f"{name} is trainable \n")
                # print(f'{name} is trainable')
                param.requires_grad = True
                break

    # Load pre-trained motion module
    unet_state_dict = unet.state_dict().keys()
    pretrained_motion_module = torch.load(pretrained_motion_module_path)
    for name, param in zip(pretrained_motion_module.keys(), pretrained_motion_module.values()):
        if name in unet_state_dict:
            unet.state_dict()[name].copy_(param)
            # print(f"{name} weight replace")

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * num_processes

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = ValidationPipeline(
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=noise_scheduler,
        ).to(local_rank)
    else:
        validation_pipeline = ValidationPipeline(
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=noise_scheduler,
        ).to(local_rank)
    validation_pipeline.enable_vae_slicing()

    # DDP wrapper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    # motion_module_trainable = False
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch["text"] = [
                    name if random.random() > cfg_random_null_text_ratio else "" for name in batch["text"]
                ]

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, _ = batch["pixel_values"].cpu(), batch["text"]

            ### >>>> Training >>>> ###

            # Convert videos to latent space, sampling from video
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            # scores (b f)  cond_frames(b f)
            scores = batch["score"]
            scores = torch.stack(scores)
            cond_frames = batch["cond_frames"]

            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

                # construct conditions for conditional module in PIA: affinity + conditional latent
                pixel_values = rearrange(pixel_values, "(b f) c h w -> b f c h w", f=video_length)
                pixel_values = pixel_values / 2.0 + 0.5
                pixel_values *= 255

                mask = torch.zeros((latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]))
                masked_image = torch.zeros_like(latents)

                is_cond = random.random()
                rand_size = latents.shape[0]
                if is_cond > cond_prob:
                    for rs in range(rand_size):
                        video_shape = [pixel_values.shape[0], pixel_values.shape[1]]
                        mask_coef = prepare_mask_coef_by_score(
                            video_shape,
                            cond_frame_idx=cond_frames,
                            statistic=statistic,
                            score=torch.tensor(scores).unsqueeze(0),
                        )
                        for f in range(video_length):
                            mask[rs, :, f, :, :] = mask_coef[rs, f]
                            masked_image[rs, :, f, :, :] = latents[rs, :, cond_frames[rs], :, :].clone()
                else:
                    masked_image = torch.zeros_like(latents)
                    mask = torch.zeros((latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]))

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch["text"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, mask, masked_image, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss = loss / gradient_accumulation_steps
            """if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad()"""

            # Backpropagate, using accumulate gradient if you have limited GPUs
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)

                # Calculate the gradient norm
                if (step + 1) % gradient_accumulation_steps == 0:
                    if isinstance(unet.parameters(), torch.Tensor):
                        params = [unet.parameters()]
                        grads = [p.grad for p in params if p.grad is not None]
                    else:
                        grads = [p.grad for p in unet.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in grads]), 2.0)

                """ <<< gradient clipping <<< """
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)

                # Calculate the gradient norm
                if (step + 1) % gradient_accumulation_steps == 0:
                    if isinstance(unet.parameters(), torch.Tensor):
                        params = [unet.parameters()]
                        grads = [p.grad for p in params if p.grad is not None]
                    else:
                        grads = [p.grad for p in unet.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in grads]), 2.0)

                """ <<< gradient clipping <<< """
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()

            if (step + 1) % gradient_accumulation_steps == 0:
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1 * gradient_accumulation_steps)

            global_step += 1

            ### <<<< Training <<<< ###

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb and ((step + 1) % gradient_accumulation_steps == 0):
                wandb.log({"gradient_norm": total_norm.item()}, step=global_step)

            # Save checkpoint and Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                height = (
                    train_data.sample_size[0]
                    if not isinstance(train_data.sample_size, int)
                    else train_data.sample_size
                )
                width = (
                    train_data.sample_size[1]
                    if not isinstance(train_data.sample_size, int)
                    else train_data.sample_size
                )

                prompts = (
                    validation_data.prompts[:2]
                    if global_step < 1000 and (not image_finetune)
                    else validation_data.prompts
                )

                # validate both for i2v and t2v
                for idx, prompt in enumerate(prompts):
                    use_image = False
                    if not image_finetune:
                        if idx < 2:
                            use_image = idx + 1
                        else:
                            use_image = False
                        sample = validation_pipeline(
                            prompt,
                            use_image=use_image,
                            generator=generator,
                            video_length=train_data.sample_n_frames,
                            height=512,
                            width=512,
                            **validation_data,
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)

                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            num_inference_steps=validation_data.get("num_inference_steps", 25),
                            guidance_scale=validation_data.get("guidance_scale", 8.0),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)

                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)

                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")

                save_path = os.path.join(output_dir, "checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                inpaint_ckpt = state_dict["state_dict"]
                trained_ckpt = {}
                for key, value in zip(inpaint_ckpt.keys(), inpaint_ckpt.values()):
                    new_key = key.replace("module.", "")
                    trained_ckpt[new_key] = value
                if step == len(train_dataloader) - 1:
                    torch.save(trained_ckpt, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(trained_ckpt, os.path.join(save_path, f"checkpoint{step+1}.ckpt"))

                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                logging.info(f"(global_step: {global_step}) loss: {loss.detach().item()}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="slurm")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
