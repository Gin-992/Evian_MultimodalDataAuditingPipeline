from dataclasses import dataclass


@dataclass(frozen=True)
class LowQualityConfig:
    analysis_probability_knowledge: float = 0.8
    analysis_probability_reasoning: float = 0.6
    default_batch_size: int = 64


@dataclass(frozen=True)
class ScoringConfig:
    default_batch_size: int = 16
    neutral_no_content_score_text: str = "Score: 2\nExplanation: No content detected."


LOW_QUALITY_CONFIG = LowQualityConfig()
SCORING_CONFIG = ScoringConfig()