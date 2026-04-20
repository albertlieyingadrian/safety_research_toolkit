from .torch_hooks import (
    steering_hook,
    get_hidden_states,
    collect_qa_hidden_states,
    get_steering_vector,
    generate_with_steering,
)
from .utils import (
    remove_all_hooks,
    clean_memory,
    get_hidden_states_for_each_sentence_end_position,
    get_hidden_states_for_every_nth_token,
)

# nnsight-based exports — available only when nnsight is installed
try:
    from .nnsight_hooks import (
        extract_activations_at_token,
        extract_activations_all_layers,
        generate_contrastive_vectors,
        load_nnsight_model,
    )
    _NNSIGHT_EXPORTS = [
        "extract_activations_at_token",
        "extract_activations_all_layers",
        "generate_contrastive_vectors",
        "load_nnsight_model",
    ]
except ImportError:
    _NNSIGHT_EXPORTS = []

__all__ = [
    "steering_hook",
    "get_hidden_states",
    "collect_qa_hidden_states",
    "get_steering_vector",
    "generate_with_steering",
    "remove_all_hooks",
    "clean_memory",
    "get_hidden_states_for_each_sentence_end_position",
    "get_hidden_states_for_every_nth_token",
] + _NNSIGHT_EXPORTS
