import torch
import torch.nn as nn
import re
from typing import Union, List, Dict, Callable, Optional, Any, Tuple
import functools
from dataclasses import dataclass, field

class TensorName:
    """
    Constants for tensor names to avoid hardcoding strings.
    """
    INPUT = 'input'
    OUTPUT = 'output'
    WEIGHT = 'weight'
    BIAS = 'bias'
    SCALE = 'scale'
    ZERO = 'zero'
    MAXQ = 'maxq'

@dataclass
class TargetSpec:
    """
    Specification for capturing tensors from modules.
    
    Args:
        modules: Modules to hook. Can be:
            - List of module names (str)
            - List of module types (type)
            - Regex string (str)
            - Filter function (Callable[[str, nn.Module], bool])
        dynamic_targets: List of attributes to capture every forward pass (e.g., 'input', 'output', 'scale').
        static_targets: List of attributes to capture only once (e.g., 'weight', 'bias').
    """
    modules: Union[List[str], List[type], str, Callable]
    dynamic_targets: List[str] = field(default_factory=lambda: [TensorName.INPUT, TensorName.OUTPUT])
    static_targets: List[str] = field(default_factory=lambda: [TensorName.WEIGHT, TensorName.BIAS])

def default_process_fn(tensor: Any) -> Any:
    """
    Default processing function: detach and move to CPU.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    return tensor

def catch_tensors(
    model: nn.Module,
    specs: Union[TargetSpec, List[TargetSpec]],
    max_samples: int = 100,
    data_container: Optional[Dict] = None,
    process_fn: Optional[Callable] = None
) -> Tuple[List[Any], Dict]:
    """
    Registers hooks to capture tensors from the model during forward pass based on provided specifications.

    Args:
        model: The model to analyze.
        specs: A TargetSpec or a list of TargetSpec objects defining what to capture.
        max_samples: Max samples for dynamic tensors.
        data_container: Dictionary to store results. If None, a new one is created.
        process_fn: Function to process captured tensors. Default is .detach().cpu().
    
    Returns:
        A tuple containing:
        - List of hook handles. Call handle.remove() to stop capturing.
        - The data_container dictionary with captured data.
    """
    if data_container is None:
        data_container = {}
    
    if process_fn is None:
        process_fn = default_process_fn

    if isinstance(specs, TargetSpec):
        specs = [specs]

    # Helper to determine if a module matches a spec's modules filter
    def is_target(name: str, module: nn.Module, modules_filter: Union[List[str], List[type], str, Callable]) -> bool:
        if isinstance(modules_filter, list):
            if len(modules_filter) > 0 and isinstance(modules_filter[0], str):
                return name in modules_filter
            if len(modules_filter) > 0 and isinstance(modules_filter[0], type):
                return isinstance(module, tuple(modules_filter))
        elif isinstance(modules_filter, str):
            return re.search(modules_filter, name) is not None
        elif callable(modules_filter):
            return modules_filter(name, module)
        return False

    handles = []

    def hook_fn(module, input, output, name, dynamic_targets, static_targets):
        # Initialize storage for this module if not exists
        if name not in data_container:
            data_container[name] = {}

        # 1. Capture Static Targets (Only once)
        for target in static_targets:
            if target not in data_container[name]:
                if hasattr(module, target):
                    val = getattr(module, target)
                    if val is not None:
                        data_container[name][target] = process_fn(val)

        # 2. Capture Dynamic Targets (Every run, up to max_samples)
        current_sample_count = 0
        # Check current count based on the first dynamic target found
        for target in dynamic_targets:
            if target in data_container[name]:
                current_sample_count = len(data_container[name][target])
                break
        
        if current_sample_count < max_samples:
            for target in dynamic_targets:
                if target not in data_container[name]:
                    data_container[name][target] = []
                
                val = None
                if target == TensorName.INPUT:
                    val = input[0] if isinstance(input, tuple) and len(input) > 0 else input
                elif target == TensorName.OUTPUT:
                    val = output
                elif hasattr(module, target):
                    # Capture module attributes (buffers/params) dynamically
                    val = getattr(module, target)
                
                if val is not None:
                    data_container[name][target].append(process_fn(val))

    for name, module in model.named_modules():
        # Determine targets for this module based on all specs
        module_dynamic_targets = set()
        module_static_targets = set()
        
        matched = False
        for spec in specs:
            if is_target(name, module, spec.modules):
                matched = True
                module_dynamic_targets.update(spec.dynamic_targets)
                module_static_targets.update(spec.static_targets)
        
        if not matched:
            continue

        dynamic_targets = list(module_dynamic_targets)
        static_targets = list(module_static_targets)

        # Register hook
        handle = module.register_forward_hook(
            functools.partial(hook_fn, name=name, dynamic_targets=dynamic_targets, static_targets=static_targets)
        )
        handles.append(handle)
            
        # Attempt to capture static targets immediately (before forward)
        if name not in data_container:
            data_container[name] = {}
        
        for target in static_targets:
            if target not in data_container[name]:
                if hasattr(module, target):
                    val = getattr(module, target)
                    if val is not None:
                        data_container[name][target] = process_fn(val)

    return handles, data_container
