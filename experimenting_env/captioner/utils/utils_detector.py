from experimenting_env.agents.goal_exploration import ObjectDetectorGTEnv


def select_object_detector(cfg):
    arch_name = cfg.arch_name
    assert arch_name.lower() in ["gt"], "Currently, only 'gt' architectures are supported."
    if arch_name.lower() == "gt":
        return ObjectDetectorGTEnv()
    return None