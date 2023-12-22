# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from animatediff.pipelines import I2VPipeline
from animatediff.utils.util import save_videos_grid, preprocess_img


def seed_everything(seed):
    import random

    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    functional_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--config",   type=str, default='configs/test.yaml')
    parser.add_argument("--magnitude", type=int, default=None, choices=[0, 1, 2, -1, -2, -3]) # negative is for style transfer
    functional_group.add_argument('--loop', action='store_true')
    functional_group.add_argument("--style_transfer", action="store_true")
    args = parser.parse_args()

    config      = OmegaConf.load(args.config)
    base_config = OmegaConf.load(config.base)
    config      = OmegaConf.merge(base_config, config)

    if args.magnitude is not None:
        config.validation_data.mask_sim_range = [args.magnitude]

    if args.style_transfer:
        config.validation_data.mask_sim_range = [-1 * magnitude - 1 if magnitude >= 0 else magnitude for magnitude in config.validation_data.mask_sim_range]
    elif args.loop:
        config.validation_data.mask_sim_range = [magnitude + 3 if magnitude >= 0 else magnitude for magnitude in config.validation_data.mask_sim_range]

    os.makedirs(config.validation_data.save_path, exist_ok=True)
    folder_num = len(os.listdir(config.validation_data.save_path))
    target_dir = f'{config.validation_data.save_path}/{folder_num}/'

    # prepare paths and pipeline
    base_model_path = config.pretrained_model_path
    unet_path = config.generate.model_path
    dreambooth_path = config.generate.db_path
    if config.generate.use_lora:
        lora_path = config.generate.get('lora_path', None)
        lora_alpha = config.generate.get('lora_alpha', 0)
    else:
        lora_path = None
        lora_alpha = 0
    validation_pipeline = I2VPipeline.build_pipeline(
        config,
        base_model_path,
        unet_path,
        dreambooth_path,
        lora_path,
        lora_alpha,
    )
    generator       = torch.Generator(device='cuda')
    generator.manual_seed(config.generate.global_seed)

    global_inf_num = 0

    # if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)

    # print(" >>> Begin test >>>")
    print(f'using unet      : {unet_path}')
    print(f'using DreamBooth: {dreambooth_path}')
    print(f'using Lora      : {lora_path}')

    sim_ranges = config.validation_data.mask_sim_range
    if isinstance(sim_ranges, int):
        sim_ranges = [sim_ranges]

    OmegaConf.save(config, os.path.join(target_dir, "config.yaml"))
    generator.manual_seed(config.generate.global_seed)
    seed_everything(config.generate.global_seed)

    # load image
    img_root = config.validation_data.validation_input_path
    input_name = config.validation_data.input_name
    if os.path.exists(os.path.join(img_root, f'{input_name}.jpg')):
        image_name = os.path.join(img_root, f'{input_name}.jpg')
    elif os.path.exists(os.path.join(img_root, f'{input_name}.png')):
        image_name = os.path.join(img_root, f'{input_name}.png')
    else:
        raise ValueError(f"image_name should be .jpg or .png")
    # image = np.array(Image.open(image_name))
    image, gen_height, gen_width  = preprocess_img(image_name)
    config.generate.sample_height = gen_height
    config.generate.sample_width  = gen_width

    for sim_range in sim_ranges:
        print(f"using sim_range : {sim_range}")
        config.validation_data.mask_sim_range = sim_range
        prompt_num = 0
        for prompt, n_prompt in zip(config.prompts, config.n_prompt):
            print(f"using n_prompt  : {n_prompt}")
            prompt_num     += 1
            for single_prompt in prompt:
                print(f" >>> Begin test {global_inf_num} >>>")
                global_inf_num += 1
                image_path = ''
                sample = validation_pipeline(
                    image=image,
                                prompt=single_prompt,
                                generator       = generator,
                                # global_inf_num  = global_inf_num,
                                video_length    = config.generate.video_length,
                                height          = config.generate.sample_height,
                                width           = config.generate.sample_width,
                                negative_prompt = n_prompt,
                                mask_sim_template_idx = config.validation_data.mask_sim_range,
                                **config.validation_data,
                            ).videos
                save_videos_grid(sample, target_dir + f"{global_inf_num}_sim_{sim_range}.gif")
                print(f" <<< test {global_inf_num} Done <<<")
    print(" <<< Test Done <<<")
