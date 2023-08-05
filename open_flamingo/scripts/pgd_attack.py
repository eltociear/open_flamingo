import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(1, "/mmfs1/home/irenagao/gscratch/open_flamingo/")


def load_model(
    checkpoint_path: str,
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    lang_encoder_path: str = "anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path: str = "anas-awadalla/mpt-1b-redpajama-200b",
):
    from open_flamingo import create_model_and_transforms

    model, processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path,
        clip_vision_encoder_pretrained,
        lang_encoder_path,
        tokenizer_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model = model.to("cuda") if torch.cuda.is_available() else model
    model.eval()
    tokenizer.padding_side = "left"
    return model, processor, tokenizer


def projected_gradient_descent(
    model,
    vision_x: torch.Tensor,
    vision_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    epsilon=0.01,
    alpha=0.01,
    num_steps=100,
):
    assert vision_x.ndim == 6

    # Define the loss function (you may need to adjust this based on your specific task)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (use any optimizer you prefer)
    optimizer = optim.SGD([vision_x[vision_mask]], lr=alpha)

    # Concat the target_tokens to the input_ids
    _input_ids = torch.cat([input_ids, target_tokens], dim=1)
    _attention_mask = torch.cat([attention_mask, torch.ones_like(target_tokens)], dim=1)

    for _ in range(num_steps):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass to get the model's logits
        logits = model(
            vision_x=vision_x,
            lang_x=_input_ids,
            attention_mask=_attention_mask,
        ).logits

        # Calculate the loss (comparing the model's output to the target token)
        loss = criterion(logits[:, ], target_token_tensor)

        # Backpropagation to compute gradients
        loss.backward()

        # Update the perturbation using the gradients
        with torch.no_grad():
            input_image_grad = input_image.grad.data
            perturbation = alpha * torch.sign(input_image_grad)
            input_image.data = input_image + perturbation

            # Project the perturbation onto an L2 ball with radius epsilon
            perturbation_norm = torch.norm(
                input_image.data - torch.tensor(input_image, requires_grad=True)
            )
            if perturbation_norm > epsilon:
                input_image.data = (
                    input_image
                    + epsilon
                    * (input_image.data - torch.tensor(input_image, requires_grad=True))
                    / perturbation_norm
                )

    # Convert the adversarial image back to numpy
    adversarial_image = input_image.squeeze(0).detach().numpy()

    # Return the adversarial image
    return adversarial_image
