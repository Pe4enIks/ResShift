#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, math, random

import numpy as np

import torch
import torch.nn.functional as F

from .utils.util_image import ImageSpliterTh

from .utils import util_net
from .utils import util_image
from .utils import util_common


class BaseSampler:
    def __init__(
        self,
        configs,
        sf=None,
        use_fp16=False,
        chop_size=128,
        chop_stride=128,
        chop_bs=1,
        desired_min_size=64,
        seed=10000,
        device="cpu",
        package_root=None,
    ):
        """
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        """
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf
        self.device = device
        self.package_root = package_root

        if self.device != "cpu":
            self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert (
            num_gpus == 1
        ), "Please assign one available GPU using CUDA_VISIBLE_DEVICES!"

        self.num_gpus = num_gpus
        self.rank = int(os.environ["LOCAL_RANK"]) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.device == "cpu":
            print(log_str)
            return

        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f"Building the diffusion model with length: {self.configs.diffusion.params.steps}..."
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(
            self.package_root, self.configs.diffusion
        )
        model = util_common.instantiate_from_config(
            self.package_root, self.configs.model
        ).to(self.device)
        ckpt_path = self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f"Loading Diffusion model from {ckpt_path}...")
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f"Loading AutoEncoder model from {ckpt_path}...")
            autoencoder = util_common.instantiate_from_config(
                self.package_root, self.configs.autoencoder
            ).to(self.device)
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        if self.device != "cpu":
            state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        else:
            state = torch.load(ckpt_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        util_net.reload_model(model, state)


class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False):
        """
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        """
        if noise_repeat:
            self.setup_seed()

        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode="reflect")
        else:
            flag_pad = False

        model_kwargs = (
            {
                "lq": y0,
            }
            if self.configs.model.params.cond_lq
            else None
        )
        results = self.base_diffusion.p_sample_loop(
            y=y0,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            noise_repeat=noise_repeat,
            clip_denoised=(self.autoencoder is None),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
        )  # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, : ori_h * self.sf, : ori_w * self.sf]

        return results.clamp_(-1.0, 1.0)

    def _process_per_image(self, im_lq_tensor, noise_repeat=False):
        """
        Input:
            im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            im_sr: h x w x c, numpy array, [0,1], RGB
        """

        if (
            im_lq_tensor.shape[2] > self.chop_size
            or im_lq_tensor.shape[3] > self.chop_size
        ):
            im_spliter = ImageSpliterTh(
                im_lq_tensor,
                self.chop_size,
                stride=self.chop_stride,
                sf=self.sf,
                extra_bs=self.chop_bs,
            )
            for im_lq_pch, index_infos in im_spliter:
                im_sr_pch = self.sample_func(
                    (im_lq_pch - 0.5) / 0.5,
                    noise_repeat=noise_repeat,
                )  # 1 x c x h x w, [-1, 1]
                im_spliter.update(im_sr_pch, index_infos)
            im_sr_tensor = im_spliter.gather()
        else:
            im_sr_tensor = self.sample_func(
                (im_lq_tensor - 0.5) / 0.5,
                noise_repeat=noise_repeat,
            )  # 1 x c x h x w, [-1, 1]

        im_sr_tensor = im_sr_tensor * 0.5 + 0.5
        return im_sr_tensor

    def inference_single(self, im_lq: np.ndarray):
        # 1 x c x h x w
        im_lq_tensor = util_image.img2tensor(im_lq).to(self.device) / 255.0
        im_sr_tensor = self._process_per_image(im_lq_tensor, False)
        im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
        return im_sr


if __name__ == "__main__":
    pass
