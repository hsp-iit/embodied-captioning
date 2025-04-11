
class Configuration:
    def __init__(self, arch_name=None, model_name=None, checkpoint_name=None, height=None, width=None):
        self.captioner = CaptionerField(arch_name=arch_name, model_name=model_name, checkpoint_name=checkpoint_name, height=height, width=width)

class CaptionerField:
    def __init__(self, arch_name=None, model_name=None, checkpoint_name=None, height=None, width=None):
        self.arch_name = arch_name
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.height = height
        self.width = width