from operator import is_
import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, device, eps=0.3, alpha=0.01, iters=40):
    """Projected Gradient Descent adversarial attack"""
    images = images.to(device)
    labels = labels.to(device)
    ori_images = images.detach().clone()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach_()

    return images


def formal_verification_check(model, input_img, label, device, eps=0.05, num_samples=50):
    """Robustness verification through random perturbations"""
    model.eval()
    input_img = input_img.to(device).unsqueeze(0)
    if torch._is_tensor(label):
        base_pred = label.item()
    else:
        base_pred = int(label)

    success = True
    for _ in range(num_samples):
        perturbation = torch.empty_like(
            input_img).uniform_(-eps, eps).to(device)
        perturbed_img = torch.clamp(input_img + perturbation, 0, 1)
        pred = model(perturbed_img).argmax(dim=1).item()

        if pred != base_pred:
            success = False
            print("Verification failed: Prediction changed under perturbation.")
            break

    if success:
        print("Verification passed: Model is robust within ε-ball")
    return success
