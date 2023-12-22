import json
import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from omegaconf import OmegaConf
from PIL import Image

from animatediff.pipelines import I2VPipeline
from animatediff.utils.util import save_videos_grid

sample_idx = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='example/config/base.yaml')
parser.add_argument('--server-name', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--share', action='store_true')

parser.add_argument('--save-path', default='samples')

args = parser.parse_args()


N_PROMPT = ('wrong white balance, dark, sketches,worst quality,low quality, '
            'deformed, distorted, disfigured, bad eyes, wrong lips, '
            'weird mouth, bad teeth, mutated hands and fingers, bad anatomy,'
            'wrong anatomy, amputation, extra limb, missing limb, '
            'floating,limbs, disconnected limbs, mutation, ugly, disgusting, '
            'bad_pictures, negative_hand-neg')


def preprocess_img(img_np, max_size: int = 512):

    ori_image = Image.fromarray(img_np).convert('RGB')

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
        in_image = ori_image

    in_image = ori_image.resize((in_width, in_height))
    in_image_np = np.array(in_image)
    return in_image_np, in_height, in_width


class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.personalized_model_dir = os.path.join(
            self.basedir, "models", "DreamBooth_LoRA")
        self.ip_adapter_dir = os.path.join(
            self.basedir, "models", "IP_Adapter")
        self.savedir = os.path.join(
            self.basedir, args.save_path, datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_personalized_model()

        self.pipeline = None

        self.inference_config = OmegaConf.load(args.config)
        self.stable_diffusion_dir = self.inference_config.pretrained_model_path
        self.pia_path = self.inference_config.generate.model_path
        self.loaded = False

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(
            self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [
            os.path.basename(p) for p in personalized_model_list]

    def get_ip_apdater_folder(self):
        file_list = os.listdir(self.ip_adapter_dir)
        if not file_list:
            return False

        if not 'ip-adapter_sd15.bin' not in file_list:
            print('Cannot find "ip-adapter_sd15.bin" '
                  f'under {self.ip_adapter_dir}')
            return False
        if not 'image_encoder' not in file_list:
            print(f'Cannot find "image_encoder" under {self.ip_adapter_dir}')
            return False

        return True

    def load_model(self,
                   dreambooth_path=None,
                   lora_path=None,
                   lora_alpha=1.0,
                   enable_ip_adapter=True):
        gr.Info('Start Load Models...')
        print('Start Load Models...')

        if lora_path and lora_path.upper() != 'NONE':
            lora_path = osp.join(self.personalized_model_dir, lora_path)
        else:
            lora_path = None

        if dreambooth_path and dreambooth_path.upper() != 'NONE':
            dreambooth_path = osp.join(
                self.personalized_model_dir, dreambooth_path)
        else:
            dreambooth_path = None

        if enable_ip_adapter:
            if not self.get_ip_apdater_folder():
                print('Load IP-Adapter from remote.')
                ip_adapter_path = 'h94/IP-Adapter'
            else:
                ip_adapter_path = self.ip_adapter_dir
        else:
            ip_adapter_path = None

        self.pipeline = I2VPipeline.build_pipeline(
            self.inference_config,
            self.stable_diffusion_dir,
            unet_path=self.pia_path,
            dreambooth_path=dreambooth_path,
            lora_path=lora_path,
            lora_alpha=lora_alpha,
            ip_adapter_path=ip_adapter_path)
        gr.Info('Load Finish!')
        print('Load Finish!')
        self.loaded = True

        return 'Load'

    def animate(
        self,
        init_img,
        motion_scale,
        prompt_textbox,
        negative_prompt_textbox,
        sampler_dropdown,
        sample_step_slider,
        length_slider,
        cfg_scale_slider,
        seed_textbox,
        ip_adapter_scale,
        progress=gr.Progress(),
    ):
        if not self.loaded:
            raise gr.Error(f"Please load model first!")

        if seed_textbox != -1 and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()
        init_img, h, w = preprocess_img(init_img)
        sample = self.pipeline(
            image=init_img,
            prompt=prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            num_inference_steps=sample_step_slider,
            guidance_scale=cfg_scale_slider,
            width=w,
            height=h,
            video_length=16,
            mask_sim_template_idx=motion_scale,
            ip_adapter_scale=ip_adapter_scale,
            progress_fn=progress,
        ).videos

        save_sample_path = os.path.join(
            self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)

        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": w,
            "height": h,
            "video_length": length_slider,
            "seed": seed,
            "motion": motion_scale,
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")

        return save_sample_path


controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        motion_idx = gr.State(0)
        gr.HTML(
            "<div align='center'><font size='7'> <img src=\"file/pia.png\" style=\"height: 72px;\"/ > Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models </font></div>"
        )
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='5'><a href='https://pi-animator.github.io/'>Project Page</a> &ensp;"  # noqa
                "<a href='https://arxiv.org/abs/2312.13964/'>Paper</a> &ensp;"
                "<a href='https://github.com/open-mmlab/pia'>Code</a> &ensp;"  # noqa
                "<a href='https://openxlab.org.cn/apps/detail/zhangyiming/PiaPia'>Demo</a> </font></div>"  # noqa
            )

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints (select pretrained model path first).
                """
            )
            with gr.Row():

                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )

                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )

                lora_alpha_slider = gr.Slider(
                    label="LoRA alpha", value=0, minimum=0, maximum=2, interactive=True)

                personalized_refresh_button = gr.Button(
                    value="\U0001F503", elem_classes="toolbutton")

                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        controller.personalized_model_list,
                        ["none"] + controller.personalized_model_list
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[
                                                  base_model_dropdown, lora_model_dropdown])

            load_model_button = gr.Button(value='Load')
            load_model_button.click(
                fn=controller.load_model,
                inputs=[
                    base_model_dropdown,
                    lora_model_dropdown,
                    lora_alpha_slider,
                ],
                outputs=[load_model_button])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for PIA.
                """
            )

            prompt_textbox = gr.Textbox(label="Prompt", lines=2)
            negative_prompt_textbox = gr.Textbox(
                value=N_PROMPT,
                label="Negative prompt", lines=1)

            with gr.Row(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        init_img = gr.Image(label='Input Image')

                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(label="Sampling method", choices=list(
                            scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(
                            label="Sampling steps", value=25, minimum=10, maximum=100, step=1)

                    length_slider = gr.Slider(
                        label="Animation length", value=16, minimum=8, maximum=24, step=1)
                    cfg_scale_slider = gr.Slider(
                        label="CFG Scale", value=7.5, minimum=0, maximum=20)
                    motion_scale_silder = gr.Slider(
                        label='Motion Scale', value=motion_idx.value, step=1, minimum=0, maximum=2)
                    ip_adapter_scale = gr.Slider(
                        label='IP-Apdater Scale', value=0.0, minimum=0, maximum=1)

                    def GenerationMode(motion_scale_silder, option):
                        if option == 'Animation':
                            motion_idx = motion_scale_silder
                        elif option == 'Style Transfer':
                            motion_idx = motion_scale_silder * -1 - 1
                        elif option == 'Loop Video':
                            motion_idx = motion_scale_silder + 3
                        return motion_idx

                    with gr.Row():
                        style_selection = gr.Radio(
                            ['Animation', 'Style Transfer', 'Loop Video'],
                            label='Generation Mode', value='Animation')
                        style_selection.change(
                            fn=GenerationMode,
                            inputs=[motion_scale_silder, style_selection],
                            outputs=[motion_idx]
                        )
                        motion_scale_silder.change(
                            fn=GenerationMode,
                            inputs=[motion_scale_silder, style_selection],
                            outputs=[motion_idx]
                        )

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button = gr.Button(
                            value="\U0001F3B2", elem_classes="toolbutton")
                    seed_button.click(
                        fn=lambda x: random.randint(1, 1e8),
                        outputs=[seed_textbox],
                        queue=False
                    )

                    generate_button = gr.Button(
                        value="Generate", variant='primary')

                result_video = gr.Video(
                    label="Generated Animation", interactive=False)

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    init_img,
                    motion_idx,
                    prompt_textbox,
                    negative_prompt_textbox,
                    sampler_dropdown,
                    sample_step_slider,
                    length_slider,
                    cfg_scale_slider,
                    seed_textbox,
                    ip_adapter_scale,
                ],
                outputs=[result_video]
            )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(server_name=args.server_name,
                server_port=args.port, share=args.share, allowed_paths=['pia.png'])
