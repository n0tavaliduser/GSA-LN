from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

@MODEL_REGISTRY.register()
class TriFANetModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)
