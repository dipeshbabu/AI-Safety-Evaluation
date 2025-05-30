import torch
import numpy as np


def integrated_gradients(model, input_tensor, target_label, baseline, steps=50, device="cpu"):
    """Enhanced integrated gradients with uncertainty tracking"""
    baseline = baseline.to(device)
    input_tensor = input_tensor.to(device)

    scaled_inputs = [baseline + (float(i)/steps)*(input_tensor - baseline)
                     for i in range(steps+1)]

    grads = []
    model.eval()
    for inp in scaled_inputs:
        inp = inp.clone().detach().requires_grad_(True)
        output = model(inp)
        score = output[0, target_label]
        model.zero_grad()
        score.backward(retain_graph=True)
        grads.append(inp.grad.detach())

    grads = torch.stack(grads)
    avg_grad = grads.mean(dim=0)
    grad_variance = grads.var(dim=0)

    return (input_tensor - baseline) * avg_grad, grad_variance


def dynamic_baseline_selection(model, input_tensor, target_label,
                               candidate_baselines, steps=50, device="cpu"):
    """Adaptive baseline selection based on gradient stability"""
    best_ig, best_baseline, min_variance = None, None, float('inf')

    for baseline in candidate_baselines:
        ig, variance = integrated_gradients(
            model, input_tensor, target_label,
            baseline.to(device), steps, device
        )
        total_var = variance.sum().item()

        if total_var < min_variance:
            min_variance = total_var
            best_baseline = baseline
            best_ig = ig

    return best_baseline, best_ig


def compute_saliency_map(model, input_tensor, target_label, device):
    """Higher-order gradient saliency mapping"""
    model.eval().to(device)
    input_tensor = input_tensor.to(device).requires_grad_(True)

    # First-order gradient
    output = model(input_tensor)
    score = output[0, target_label]
    model.zero_grad()
    score.backward(retain_graph=True)
    grad_1 = input_tensor.grad.clone()

    # Second-order approximation
    eps = 1e-2
    grad_2 = torch.zeros_like(input_tensor)
    for i in range(input_tensor.numel()):
        perturb = torch.zeros_like(input_tensor)
        perturb.view(-1)[i] = eps
        output_perturbed = model(input_tensor + perturb)
        grad_2.view(-1)[i] = (output_perturbed[0,
                                               target_label] - score).item() / eps

    return (grad_1.abs() + grad_2.abs()).detach().squeeze().cpu().numpy()
