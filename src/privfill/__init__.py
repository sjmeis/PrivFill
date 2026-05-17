from enum import Enum
from .main import PrivFill, PrivFillDPBart, PrivFillDP

class SupportedModels(Enum):
    FLAN_T5_BASE = "sjmeis/flan-t5-base-infill-combined"
    FLAN_T5_LARGE = "sjmeis/flan-t5-large-infill-combined"
    BART_LARGE = "sjmeis/bart-large-infill-combined"

def load_pipeline(model_choice: SupportedModels, DP: bool = False, **kwargs):
    """
    Loads the appropriate privatization engine based on model choice and DP toggle.
    
    Args:
        model_choice (SupportedModels): The chosen model from the Enum.
        dp (bool): If True, applies the model's Differential Privacy mechanism.
                   If False, falls back to the standard PrivFill wrapper.
    """
    if not isinstance(model_choice, SupportedModels):
        raise ValueError(
            f"Invalid model choice. Please choose an option from privfill.SupportedModels. "
            f"Available choices: {list(SupportedModels.__members__.keys())}"
        )

    checkpoint = model_choice.value

    if DP:
        if model_choice == SupportedModels.BART_LARGE:
            return PrivFillDPBart(model_checkpoint=checkpoint, **kwargs)
        else:
            return PrivFillDP(model_checkpoint=checkpoint, **kwargs)
    else:
        return PrivFill(model_checkpoint=checkpoint, **kwargs)


__all__ = ["PrivFill", "PrivFillDPBart", "PrivFillDP", "SupportedModels", "load_pipeline"]