# [MODIFIED] Registers StageFiLM/BEVFusion and the vector-prior modules.
from .transformer import MapTRPerceptionTransformer
from .decoder import MapTRDecoder, DecoupledDetrTransformerDecoderLayer, InstancePointAttention, CustomTransformerLayer
from .geometry_kernel_attention import GeometrySptialCrossAttention, GeometryKernelAttention
from .height_kernel_attention import HeightKernelAttention, MSDeformableAttentionKernel
from .builder import build_fuser
from .encoder import LSSTransform
from .stagefilm_raster import BEVFusion, StageFiLMRasterEncoder
from .vector_prior import (
    MapGraphTransformer, SineContinuousPositionalEncoding,
    CrossAttentionBEV, DualVecCrossFusion)
