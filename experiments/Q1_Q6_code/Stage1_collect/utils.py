import torch


def print_device_summary(model):
    """Print device and hf_device_map summary for a model."""
    print("\n[Device Summary]")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "<unknown>"
            print(f"  cuda:{i} -> {name}")

    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict) and hf_map:
        print("- hf_device_map (first 30 entries):")
        for idx, (k, v) in enumerate(hf_map.items()):
            if idx >= 30:
                print(f"  ... ({len(hf_map) - 30} more)")
                break
            print(f"  {k} -> {v}")
    else:
        print("- hf_device_map: None")

    # primary device and dtype sampling
    try:
        first_param = next(model.parameters())
        print(f"- primary device: {first_param.device}")
        print(f"- sample dtype:  {first_param.dtype}")
    except StopIteration:
        print("- model has no parameters? (meta tensors)")


def find_moe_layers(model):
    """Traverse model to find MoE layers and return info dict."""
    moe_layer_info = {}
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        print("Warning: model.model.layers not found.")
        return moe_layer_info

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        cls_name = mlp.__class__.__name__ if mlp is not None else ""
        if mlp is not None and "Moe" in cls_name:
            num_experts = getattr(mlp, "num_experts", None)
            top_k = getattr(mlp, "top_k", 2)
            moe_layer_info[i] = {
                "num_experts": int(num_experts),
                "top_k": int(top_k),
            }
            print(f"Found MoE Layer {i}: experts={num_experts}, top_k={top_k}")

    if not moe_layer_info:
        print("Warning: No MoE layers found in the model. Is this the correct MoE model?")
    return moe_layer_info


def disable_qwen2moe_aux_loss():
    """Disable Qwen2-MoE auxiliary load-balancing loss to avoid runtime issues."""
    try:
        import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen2_moe_mod  # type: ignore

        if hasattr(qwen2_moe_mod, "load_balancing_loss_func"):

            def _noop(*args, **kwargs):
                dev = None
                for a in list(args) + list(kwargs.values()):
                    if isinstance(a, torch.Tensor):
                        dev = a.device
                        break
                return (
                    torch.tensor(0.0, device=dev)
                    if dev is not None
                    else torch.tensor(0.0)
                )

            qwen2_moe_mod.load_balancing_loss_func = _noop  # type: ignore
            print("[patch] Disabled Qwen2-MoE aux load-balancing loss")
    except Exception as e:
        print(f"[patch] Skip disabling aux loss: {e}")
