import copyreg
dict_keys_type = type({}.keys())
def _reduce_dict_keys(dk):
    return (list, (list(dk),))
copyreg.pickle(dict_keys_type, _reduce_dict_keys)

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.backbones.vovnet import VoVNet
from .models.utils import *
from .models.opt.adamw import AdamW2
from .VAD import *
