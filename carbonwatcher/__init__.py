from carbonwatcher.core.tracker import CarbonTracker, TrackingResult, EpochResult
from carbonwatcher.core.carbon import CarbonEstimator, CarbonResult
from carbonwatcher.core.energy import EnergyCalculator, EnergyResult
from carbonwatcher.core.hardware import HardwareProfiler, HardwareProfile
from carbonwatcher.core.cost import CostTracker, RunRecord
from carbonwatcher.llm.tracker import LLMTracker, LLMCallResult, LLMCallRecord
from carbonwatcher.llm.models import ModelSpec, lookup_model, list_models
from carbonwatcher.llm.tokenizer import count_tokens, count_messages
from carbonwatcher import parser

__version__ = "0.2.0"
__all__ = [
    # Core tracking
    "CarbonTracker", "TrackingResult", "EpochResult",
    # Sub-components
    "CarbonEstimator", "CarbonResult",
    "EnergyCalculator", "EnergyResult",
    "HardwareProfiler", "HardwareProfile",
    "CostTracker", "RunRecord",
    # LLM tracking
    "LLMTracker", "LLMCallResult", "LLMCallRecord",
    "ModelSpec", "lookup_model", "list_models",
    # Token counting
    "count_tokens", "count_messages",
    # Log parsing
    "parser",
    # Online wrappers (lazy — need anthropic/openai installed)
    # from carbonwatcher.llm.anthropic_wrapper import TrackedAnthropic
    # from carbonwatcher.llm.openai_wrapper import TrackedOpenAI
    # Local wrappers (lazy — need ollama/transformers/llama-cpp-python installed)
    # from carbonwatcher.llm.ollama_wrapper import TrackedOllama
    # from carbonwatcher.llm.huggingface_wrapper import TrackedHuggingFace
    # from carbonwatcher.llm.llamacpp_wrapper import TrackedLlamaCpp
]
