"""RAPD-Net model definitions."""

from .stage1_spectral_basis import Stage1SpectralBasisNet
from .stage1_unmixing import Stage1UnmixingNet
from .stage2_coefficient_residual import Stage2CoefficientResidualNet
from .stage2_dual_space import Stage2DualSpaceNet
from .stage2_frequency_reliability import (
    ChannelWiseSpectralSplitter,
    FrequencyReliabilityScreen,
    NoiseSplitter,
    SharedMSIFeatureEncoder,
)
from .stage2_multiscale_pyramid import Stage2MultiScalePyramidNet
from .stage2_physical_fusion import Stage2PhysicalFusionNet
from .stage2_srf_anchor import Stage2SRFAnchorNet
from .stage2_symmetric_frequency import (
    Stage2SymmetricFrequencyNet,
    SymmetricFrequencyReliabilityScreen,
)
from .stage3_ablation_mask_curriculum import MaskCurriculumAblationRefiner
from .stage3_dual_domain_diffusion import (
    BasisOrthogonalResidualDiffusionRefiner,
    ConditionalResidualDenoiser,
    GaussianDiffusionSchedule,
)
from .stage3_uncertainty_guided_diffusion import (
    DeterministicUncertaintyPredictor,
    LocalConditionalNoiseDenoiser,
    UncertaintyGuidedDualDomainDiffusionRefiner,
)
from .stage3_uncertainty_guided_diffusion_v2 import (
    LocalConditionalHybridDenoiser,
    UncertaintyGuidedDualDomainDiffusionRefinerV2,
)
from .stage3_uncertainty_guided_diffusion_v2_stable import (
    UncertaintyGuidedDualDomainDiffusionRefinerV2Stable,
)

__all__ = [
    "Stage1SpectralBasisNet",
    "Stage1UnmixingNet",
    "Stage2CoefficientResidualNet",
    "Stage2SRFAnchorNet",
    "Stage2DualSpaceNet",
    "Stage2SymmetricFrequencyNet",
    "Stage2MultiScalePyramidNet",
    "BasisOrthogonalResidualDiffusionRefiner",
    "ConditionalResidualDenoiser",
    "GaussianDiffusionSchedule",
    "UncertaintyGuidedDualDomainDiffusionRefiner",
    "DeterministicUncertaintyPredictor",
    "LocalConditionalNoiseDenoiser",
    "MaskCurriculumAblationRefiner",
    "UncertaintyGuidedDualDomainDiffusionRefinerV2",
    "UncertaintyGuidedDualDomainDiffusionRefinerV2Stable",
    "LocalConditionalHybridDenoiser",
    "SharedMSIFeatureEncoder",
    "ChannelWiseSpectralSplitter",
    "NoiseSplitter",
    "FrequencyReliabilityScreen",
    "SymmetricFrequencyReliabilityScreen",
    "Stage2PhysicalFusionNet",
]
