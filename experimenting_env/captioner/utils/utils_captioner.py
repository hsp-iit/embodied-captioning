from experimenting_env.captioner.models.coca.coca import CoCa
from experimenting_env.captioner.models.blip2.blip2 import BLIP2

def select_captioner(cfg):
    arch_name = cfg.arch_name
    assert arch_name.lower() in ["coca", "blip2"], "Currently, only 'coca' and 'blip2' architectures are supported."
    if arch_name.lower() == "coca":
        return CoCa(cfg)
    elif arch_name.lower() == "blip2":
        return BLIP2(cfg)
    return None
