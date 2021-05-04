import collections
import logging
import pathlib
import os
import time

import pytorch_lightning.logging as pl_logging
import omegaconf
import torch
import tqdm

import cfvpy.models
import cfvpy.rela
import cfvpy.utils
import heyhi


def _build_model(device, env_cfg, cfg, state_dict=None, half=False, jit=False):
    assert cfg is not None
    model_name = cfg.name
    kwargs = cfg.kwargs
    model_class = getattr(cfvpy.models, model_name)
    model = model_class(
        num_faces=env_cfg.num_faces, num_dice=env_cfg.num_dice, **kwargs
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if half:
        model = model.half()
    model.to(device)
    if jit:
        model = torch.jit.script(model)
    logging.info("Built a model: %s", model)
    logging.info("Params: %s", [x.dtype for x in model.parameters()])
    return model


def create_mdp_config(cfr_yaml_cfg):
    cfg_dict: dict
    if cfr_yaml_cfg is None:
        cfg_dict = {}
    else:
        cfg_dict = dict(cfr_yaml_cfg)
    logging.info(
        "Using the following kwargs to create RecursiveSolvingParams: %s", cfr_yaml_cfg
    )

    def recusive_set(cfg, cfg_dict):
        for key, value in cfg_dict.items():
            if not hasattr(cfg, key):
                raise RuntimeError(
                    f"Cannot find key {key} in {cfg}. It's either not definied"
                    " or not imposed via pybind11"
                )
            if isinstance(value, (dict, omegaconf.dictconfig.DictConfig)):
                recusive_set(getattr(cfg, key), value)
            else:
                setattr(cfg, key, value)
        return cfg

    return recusive_set(cfvpy.rela.RecursiveSolvingParams(), cfg_dict)

device = "cuda"
cfg = omegaconf.OmegaConf.load('conf/c02_selfplay/poker_sp.yaml')
print(cfg)

bin_path = "/usr4/alg504/cliao25/rebel/outputs/2021-04-24/1024/ckpt/epoch0.torchscript"
#net = _build_model(device, cfg.env, cfg.model, torch.load("outputs/2021-04-24/15-08-30/ckpt/epoch3.ckpt"))
cfg.env.random_action_prob = 0

cfvpy.rela.play_poker_dice(create_mdp_config(cfg.env), str(bin_path))

