
import torch
import torch.nn.functional as F


def generate_attention_map(model, image):
    """
    Generate a ViT attention visualization using attention rollout.

    Attention rollout combines attention weights across transformer
    layers to visualize which image patches receive attention.
    """

    attention_weights = []
    original_forwards = []
    hooks = []

    # Capture attention weights produced by a transformer layer.
    def attention_hook(module, inputs, output):
        attention_weights.append(output[1].detach())

    try:
        # Access the multi-head self-attention module in each encoder block.
        for block in model.encoder.layers:
            attention = block.self_attention

            # Save the original forward method so it can be restored.
            original_forward = attention.forward
            original_forwards.append((attention, original_forward))

            # Request attention weights instead of disabling their output.
            def forward_with_weights(*args, _original=original_forward, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return _original(*args, **kwargs)

            attention.forward = forward_with_weights

            # Register a hook to collect the returned attention weights.
            hooks.append(
                attention.register_forward_hook(attention_hook)
            )

        # Run the image through ViT to collect attention weights.
        with torch.no_grad():
            model(image)

    finally:
        # Remove hooks and restore the original model methods.
        for hook in hooks:
            hook.remove()

        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    if not attention_weights:
        raise RuntimeError("No attention weights were captured.")

    # Create an identity matrix to account for residual connections.
    result = torch.eye(
        attention_weights[0].shape[-1],
        device=image.device
    )

    # Combine attention information across transformer layers.
    for attention in attention_weights:
        attention = attention.mean(dim=1)[0]
        attention = attention + torch.eye(
            attention.shape[-1],
            device=image.device
        )

        # Normalize attention scores.
        attention = attention / attention.sum(dim=-1, keepdim=True)

        result = attention @ result

    # Extract the class token's attention to the image patches.
    mask = result[0, 1:]

    # Reshape patch attention into a 2D image grid.
    grid_size = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(1, 1, grid_size, grid_size)

    # Resize the attention map to match the original image dimensions.
    mask = F.interpolate(
        mask,
        size=image.shape[2:],
        mode="bilinear",
        align_corners=False
    )

    # Normalize attention values between 0 and 1.
    mask = mask.squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask
