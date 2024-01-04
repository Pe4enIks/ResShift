#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from .sampler import ResShiftSampler

from basicsr.utils.download_util import load_file_from_url


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument(
        "-o", "--out_path", type=str, default="./results", help="Output path."
    )
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256],
        help="Chopping forward.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="realsrx4",
        choices=["realsrx4", "bicsrx4_opencv", "bicsrx4_matlab"],
        help="Chopping forward.",
    )
    args = parser.parse_args()

    return args


def get_configs(args):
    if args.task == "realsrx4":
        configs = OmegaConf.load("./configs/realsr_swinunet_realesrgan256.yaml")
    elif args.task == "bicsrx4_opencv":
        configs = OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml")
    elif args.task == "bicsrx4_matlab":
        configs = OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml")
        configs.diffusion.params.kappa = 2.0

    # prepare the checkpoint
    ckpt_dir = Path("./weights")
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / f"resshift_{args.task}_s{args.steps}.pth"
    if not ckpt_path.exists():
        load_file_from_url(
            url=f"https://github.com/zsyOAOA/ResShift/releases/download/v1.0/{ckpt_path.name}",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
        )
    vqgan_path = ckpt_dir / f"autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = (512 - 64) * (4 // args.scale)
    elif args.chop_size == 256:
        chop_stride = (256 - 32) * (4 // args.scale)
    else:
        raise ValueError("Chop size must be in [512, 256]")
    args.chop_size *= 4 // args.scale
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    autoencoder_scale = 2 ** (len(configs.autoencoder.params.ddconfig.ch_mult) - 1)
    desired_min_size = 64 * (autoencoder_scale // args.scale)

    return configs, chop_stride, desired_min_size


def get_configs_from_global_config(root, cfg):
    scale = cfg["outscale"]
    chop_size = cfg["chop_size"]
    steps = cfg["steps"]
    task = cfg["task"]
    ckpt_dir_path = cfg["ckpt_dir"]
    path_to_resshift = cfg["resshift_location"]

    if task == "realsrx4":
        configs = OmegaConf.load(
            root / f"{path_to_resshift}/configs/realsr_swinunet_realesrgan256.yaml"
        )
    elif task == "bicsrx4_opencv":
        configs = OmegaConf.load(
            root / f"{path_to_resshift}/configs/bicubic_swinunet_bicubic256.yaml"
        )
    elif task == "bicsrx4_matlab":
        configs = OmegaConf.load(
            root / f"{path_to_resshift}/configs/bicubic_swinunet_bicubic256.yaml"
        )
        configs.diffusion.params.kappa = 2.0

    ckpt_dir = root / ckpt_dir_path
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / f"resshift_{task}_s{steps}.pth"
    base_url = "https://github.com/zsyOAOA/ResShift/releases/download/v1.0"
    if not ckpt_path.exists():
        load_file_from_url(
            url=base_url + f"/{ckpt_path.name}",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
        )
    vqgan_path = ckpt_dir / f"autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        load_file_from_url(
            url=base_url + "/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = steps
    configs.diffusion.params.sf = scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    if chop_size == 512:
        chop_stride = (512 - 64) * (4 // scale)
    elif chop_size == 256:
        chop_stride = (256 - 32) * (4 // scale)
    else:
        raise ValueError("Chop size must be in [512, 256]")
    chop_size *= 4 // scale
    print(f"Chopping size/stride: {chop_size}/{chop_stride}")

    autoencoder_scale = 2 ** (len(configs.autoencoder.params.ddconfig.ch_mult) - 1)
    desired_min_size = 64 * (autoencoder_scale // scale)

    return configs, chop_stride, desired_min_size


def main():
    args = get_parser()

    configs, chop_stride, desired_min_size = get_configs(args)

    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        desired_min_size=desired_min_size,
    )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False)


if __name__ == "__main__":
    main()
