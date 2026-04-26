# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
from PIL import Image
from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor

from flash_head.src.modules.flash_head_model import WanModelAudioProject
from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model
from flash_head.utils.utils import match_and_blend_colors_torch, resize_and_centercrop
from flash_head.utils.facecrop import process_image

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _resolve_bool_flag(env_key, configured, *, default):
    value = os.environ.get(env_key)
    if value is None:
        value = configured
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for {env_key}: {value!r}")


# use parallel vae to speedup decode/encode, only support WanVAE
USE_PARALLEL_VAE = True

def get_cond_image_dict(cond_image_path_or_dir, use_face_crop):
    def get_image(cond_image_path, use_face_crop):
        if use_face_crop:
            try:
                image = process_image(cond_image_path)
                return image
            except Exception as e:
                logger.error(f"Error processing {cond_image_path}: {e}")
        return Image.open(cond_image_path).convert("RGB")

    if os.path.isdir(cond_image_path_or_dir):
        import glob
        cond_image_list = glob.glob(os.path.join(cond_image_path_or_dir, "*.png"))
        cond_image_list.sort()
        cond_image_dict = {cond_image.split("/")[-1].split(".")[0]: get_image(cond_image, use_face_crop) for cond_image in cond_image_list}
    else:
        cond_image_dict = {cond_image_path_or_dir.split("/")[-1].split(".")[0]: get_image(cond_image_path_or_dir, use_face_crop)}
    return cond_image_dict

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class FlashHeadPipeline:
    def __init__(
        self,
        checkpoint_dir,
        model_type,
        wav2vec_dir,
        device="cuda",
        param_dtype=torch.bfloat16,
        use_usp=False,
        num_timesteps=1000,
        use_timestep_transform=True,
        compile_model=None,
        compile_vae=None,
    ):
        r"""
        Initializes the image-to-video generation model components.
        Args:
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            wav2vec_dir (`str`):
                Path to directory containing wav2vec checkpoints
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
        """
        self.param_dtype = param_dtype
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.use_usp = use_usp and dist.is_initialized()
        self.model_type = model_type
        self.use_ltx = model_type == "lite"

        if self.use_ltx:
            model_dir = os.path.join(checkpoint_dir, "Model_Lite")
            vae_dir = os.path.join(checkpoint_dir, "VAE_LTX")

            from flash_head.ltx_video.ltx_vae import LtxVAE
            self.vae = LtxVAE(
                pretrained_model_type_or_path=vae_dir,
                dtype=self.param_dtype,
                device=self.device,
            )
        else:
            vae_path = os.path.join(checkpoint_dir, "VAE_Wan/Wan2.1_VAE.pth")
            
            from flash_head.wan.modules import WanVAE
            self.vae = WanVAE(
                vae_path=vae_path,
                dtype=self.param_dtype,
                device=self.device,
                parallel=(USE_PARALLEL_VAE and self.use_usp),
            )

            if self.model_type == "pretrained":
                self.audio_guide_scale = 3.0
                model_dir = os.path.join(checkpoint_dir, "teacher")
            elif self.model_type == "pro":
                model_dir = os.path.join(checkpoint_dir, "Model_Pro")
        
        self.model = WanModelAudioProject.from_pretrained(model_dir)
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=self.param_dtype)

        self.config = self.model.config

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        resolved_compile_model = _resolve_bool_flag(
            "FLASHHEAD_COMPILE_MODEL",
            compile_model,
            default=True,
        )
        resolved_compile_vae = _resolve_bool_flag(
            "FLASHHEAD_COMPILE_VAE",
            compile_vae,
            default=True,
        )
        logger.info(
            f"FlashHead torch.compile: COMPILE_MODEL={resolved_compile_model}, "
            f"COMPILE_VAE={resolved_compile_vae}"
        )
        if resolved_compile_model:
            self.model = torch.compile(self.model)
        if resolved_compile_vae:
            if self.use_ltx:
                self.vae.model.encode = torch.compile(self.vae.model.encode)
                self.vae.model.decode = torch.compile(self.vae.model.decode)
            else:
                self.vae.encode = torch.compile(self.vae.encode)
                self.vae.decode = torch.compile(self.vae.decode)

        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)

    @torch.no_grad()
    def prepare_params(self,
                        cond_image_path_or_dir,
                        target_size,
                        frame_num,
                        motion_frames_num,
                        sampling_steps,
                        seed=None,
                        shift=5.0,
                        color_correction_strength=0.0,
                        use_face_crop=False,
                        ):
        self.cond_image_dict = get_cond_image_dict(cond_image_path_or_dir, use_face_crop)

        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num
        self.color_correction_strength = color_correction_strength

        self.target_h, self.target_w = target_size
        self.lat_h, self.lat_w = self.target_h // self.config.vae_stride[1], self.target_w // self.config.vae_stride[2]

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # prepare timesteps
        if sampling_steps == 2:
            timesteps = [1000, 500]
        elif sampling_steps == 4:
            timesteps = [1000, 750, 500, 250]
        else:
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            
        timesteps.append(0.)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
        self.timesteps = timesteps

        self.cond_image_tensor_dict = {}
        self.ref_img_latent_dict = {}
        for i, (person_name, cond_image_pil) in enumerate(self.cond_image_dict.items()):
            cond_image_tensor = resize_and_centercrop(cond_image_pil, (self.target_h, self.target_w)).to(self.device, dtype=self.param_dtype) # 1 C 1 H W
            cond_image_tensor = (cond_image_tensor / 255 - 0.5) * 2

            self.cond_image_tensor_dict[person_name] = cond_image_tensor

            video_frames = cond_image_tensor.repeat(1, 1, self.frame_num, 1, 1)
            self.ref_img_latent_dict[person_name] = self.vae.encode(video_frames) # (16, 9, 64, 64) / (128, 5, 16, 16)
            if i == 0:
                self.reset_person_name(person_name)

        return
    
    @torch.no_grad()
    def reset_person_name(self, person_name=None):
        if person_name is None or person_name not in self.cond_image_dict:
            pass
        else:
            self.person_name = person_name
        self.original_color_reference = self.cond_image_tensor_dict[self.person_name]
        self.ref_img_latent = self.ref_img_latent_dict[self.person_name]
        self.latent_motion_frames = self.ref_img_latent[:, :1].clone()

    @torch.no_grad()
    def preprocess_audio(self, speech_array, sr=16000, fps=25):
        video_length = len(speech_array) * fps / sr

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            logger.error("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb

    @torch.no_grad()
    def generate(self, audio_embedding):
        # evaluation mode
        with torch.no_grad():

            # sample videos
            noise = torch.randn(
                self.config.out_dim, 
                (self.frame_num - 1) // self.config.vae_stride[0] + 1,
                self.lat_h,
                self.lat_w,
                dtype=self.param_dtype,
                device=self.device,
                generator=self.generator)

            for i in range(len(self.timesteps)-1):
                torch.cuda.synchronize()

                noise[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

                flow_pred = self.model(
                    x=noise.unsqueeze(0),
                    timestep=self.timesteps[i],
                    context=audio_embedding,
                    y=self.ref_img_latent.unsqueeze(0),
                )[0]

                if self.model_type == "pretrained":
                    flow_pred_drop_audio = self.model(
                        x=noise.unsqueeze(0),
                        timestep=self.timesteps[i],
                        context=torch.zeros_like(audio_embedding),
                        y=self.ref_img_latent.unsqueeze(0),
                    )[0]
                    flow_pred = flow_pred_drop_audio + self.audio_guide_scale * (flow_pred - flow_pred_drop_audio)

                    # update latent
                    dt = self.timesteps[i] - self.timesteps[i + 1]
                    dt = (dt / self.num_timesteps).to(self.param_dtype)
                    noise = noise - flow_pred * dt[:, None, None, None]
                
                else:
                    # update latent
                    t_i = (self.timesteps[i][:, None, None, None] / self.num_timesteps).to(self.param_dtype)
                    t_i_1 = (self.timesteps[i+1][:, None, None, None] / self.num_timesteps).to(self.param_dtype)
                    x_0 = noise - flow_pred * t_i

                    noise = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator)

                torch.cuda.synchronize()

            noise[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            torch.cuda.synchronize()

            videos = self.vae.decode(noise)

            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(videos, self.original_color_reference, self.color_correction_strength)

        cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        self.latent_motion_frames = self.vae.encode(cond_frame)
        torch.cuda.synchronize()

        gen_video_samples = videos #[:, :, self.motion_frames_num:]

        return gen_video_samples[0].to(torch.float32)
