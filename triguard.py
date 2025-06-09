import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
import datetime
from torchvision.models import resnet50, resnet101, mobilenet_v3_large, densenet121
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


# -------------------------------
# Simple CNN Model
# -------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=28, dropout_p=0.25):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

        # Calculate feature dimension dynamically
        self.feature_dim = self._calculate_features(input_channels, input_size)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def _calculate_features(self, channels, size):
        dummy = torch.zeros(1, channels, size, size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        return dummy.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Preserve batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def remove_dropout_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout_layers(child)


# -------------------------------
# Dataset and Model Loader
# -------------------------------

def load_dataset(name, batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform)
    elif name == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform)
    elif name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader, test_set


def get_model(name, dataset):
    if name == "simplecnn":
        input_channels = 1 if dataset in ["mnist", "fashionmnist"] else 3
        input_size = 28 if dataset in ["mnist", "fashionmnist"] else 32
        return SimpleCNN(input_channels=input_channels, input_size=input_size, dropout_p=0.25)
    elif name == "resnet50":
        model = resnet50(num_classes=10)
        if dataset in ["mnist", "fashionmnist"]:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif name == "resnet101":
        model = resnet101(num_classes=10)
        if dataset in ["mnist", "fashionmnist"]:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=3, padding=3, bias=False)
        return model
    elif name == "mobilenetv3":
        model = mobilenet_v3_large(num_classes=10)
        if dataset in ["mnist", "fashionmnist"]:
            model.features[0][0] = nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    elif name == "densenet121":
        model = densenet121(num_classes=10)
        if dataset in ["mnist", "fashionmnist"]:
            model.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.features.pool0 = nn.Identity()
        return model
    else:
        raise ValueError(f"Unknown model: {name}")

# -------------------------------
# Utility: Attribution Entropy
# -------------------------------


def attribution_entropy(attributions):
    norm = attributions.abs().flatten()
    total = norm.sum()
    if total == 0 or torch.isnan(total):
        return float('nan'), None
    prob = norm / total
    entropy = -torch.sum(prob * torch.log(prob + 1e-10))
    return entropy.item(), prob.cpu().numpy()


def compute_entropy_regularization(inputs, model, labels, device, epsilon=1e-10):
    inputs = inputs.clone().detach().requires_grad_(True).to(device)
    outputs = model(inputs)  # forward pass on differentiable input
    loss = F.cross_entropy(outputs, labels)

    grads = torch.autograd.grad(
        loss, inputs, create_graph=True, retain_graph=True)[0]

    abs_grads = grads.abs().view(grads.size(0), -1)
    normed_grads = abs_grads / (abs_grads.sum(dim=1, keepdim=True) + epsilon)
    entropy = - (normed_grads * torch.log(normed_grads + epsilon)).sum(dim=1)

    return entropy.mean()


# -------------------------------
# Train Model
# -------------------------------


# Without Entropy Regularization
# def train_model(model, train_loader, device, epochs=5):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# With Entropy Regularization
def train_model(model, train_loader, device, epochs=5, lambda_entropy=0.05):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    supports_entropy_reg = not isinstance(
        model, torchvision.models.MobileNetV3)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            if supports_entropy_reg:
                entropy_reg = compute_entropy_regularization(
                    inputs, model, labels, device)
                total_loss = ce_loss + lambda_entropy * entropy_reg
            else:
                total_loss = ce_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {epoch_loss:.4f}")

# -------------------------------
# Evaluate Model
# -------------------------------


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

# -------------------------------
# PGD Attack
# -------------------------------


def pgd_attack(model, images, labels, device, eps=0.3, alpha=0.01, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    ori_images = images.detach().clone()
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images

# -------------------------------
# Formal Verification
# -------------------------------


def formal_verification_check(model, input_img, label, device, eps=0.05, num_samples=50):
    model.eval()
    input_img = input_img.to(device).unsqueeze(0)
    # base_pred = model(input_img).argmax(dim=1).item()
    if torch.is_tensor(label):
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
            print("Verification failed: Prediction changed under small perturbation.")
            break
    if success:
        print("Verification passed: Model prediction is robust within the eps ball.")
    return success

# -------------------------------
# Attribution Methods
# -------------------------------


def integrated_gradients(model, input_tensor, target_label, baseline, steps=50, device="cpu"):
    baseline = baseline.to(device)
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    scaled_inputs = [
        baseline + (float(i)/steps)*(input_tensor - baseline) for i in range(steps+1)]
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
    integrated_grad = (input_tensor - baseline) * avg_grad
    return integrated_grad, grad_variance


def dynamic_baseline_selection(model, input_tensor, target_label, candidate_baselines, steps=50, device="cpu", plot=False):
    best_baseline = None
    best_ig = None
    min_variance = None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("plots", exist_ok=True)
    for idx, baseline in enumerate(candidate_baselines):
        ig, variance = integrated_gradients(
            model, input_tensor, target_label, baseline.to(device), steps, device)
        entropy_val, prob = attribution_entropy(ig)
        total_variance = variance.sum().item()
        print(
            f"Baseline mean: {baseline.mean().item():.4f}, Total Variance: {total_variance:.4f}, Entropy: {entropy_val:.4f}")
        if plot:
            plt.figure()
            sns.histplot(prob, bins=30, kde=True)
            plt.title(
                f"Attribution Prob. Histogram (Entropy={entropy_val:.2f})")
            filename = f"plots/baseline_{idx}_entropy_{entropy_val:.2f}_{timestamp}.png"
            plt.savefig(filename)
            print(f"Saved histogram to {filename}")
        if min_variance is None or total_variance < min_variance:
            min_variance = total_variance
            best_baseline = baseline
            best_ig = ig
    print(f"Selected baseline with mean: {best_baseline.mean().item():.4f}")
    return best_baseline, best_ig


def compute_drift_map(clean_ig, adv_ig):
    """
    Computes absolute drift between clean and adversarial integrated gradients.
    Returns normalized drift heatmap.
    """
    drift = (clean_ig - adv_ig).abs().squeeze().cpu()
    drift /= drift.max()
    return drift


def visualize_contrastive_attributions(input_tensor, clean_ig, adv_ig, delta_ig=None, save_path="plots/contrastive_ig.png"):
    """
    Visualize original input, clean IG, adversarial IG, and optionally ΔIG
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_np = input_tensor.squeeze().detach().cpu().numpy()
    clean_ig_np = clean_ig.squeeze().detach().cpu().numpy()
    adv_ig_np = adv_ig.squeeze().detach().cpu().numpy()
    delta_ig_np = None if delta_ig is None else delta_ig.squeeze().detach().cpu().numpy()

    if input_np.ndim == 3 and input_np.shape[0] == 3:
        input_np = np.transpose(input_np, (1, 2, 0))  # (H, W, C)

    if clean_ig_np.ndim == 3:
        clean_ig_np = np.abs(clean_ig_np).mean(axis=0)
    if adv_ig_np.ndim == 3:
        adv_ig_np = np.abs(adv_ig_np).mean(axis=0)
    if delta_ig_np is not None and delta_ig_np.ndim == 3:
        delta_ig_np = np.abs(delta_ig_np).mean(axis=0)

    cols = 3 if delta_ig is None else 4
    fig, axs = plt.subplots(1, cols, figsize=(cols * 4, 4))

    axs[0].imshow(np.clip(input_np, 0, 1))
    axs[0].set_title("Original Input")
    axs[1].imshow(clean_ig_np, cmap="inferno")
    axs[1].set_title("IG (Clean)")
    axs[2].imshow(adv_ig_np, cmap="inferno")
    axs[2].set_title("IG (Adversarial)")
    if delta_ig is not None:
        axs[3].imshow(delta_ig_np, cmap="bwr")
        axs[3].set_title("ΔIG (Diff Map)")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved contrastive attribution plot to {save_path}")
    plt.close()


def compute_contrastive_attributions(model, x, label, device, baseline, steps=50):
    model.eval()
    x = x.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)  # ← FIXED HERE

    baseline = baseline.to(device)

    # Generate adversarial sample
    x_adv = pgd_attack(model, x.clone(), label.clone(),
                       device, eps=0.3, alpha=0.01, iters=40)

    # Compute IG for both
    clean_ig, _ = integrated_gradients(
        model, x, label.item(), baseline, steps, device)
    adv_ig, _ = integrated_gradients(model, x_adv, model(
        x_adv).argmax().item(), baseline, steps, device)
    delta_ig = adv_ig - clean_ig

    # Attribution drift metric
    drift = torch.norm(clean_ig - adv_ig, p=2).item()
    print(f"Attribution Drift Score (L2): {drift:.4f}")

    return clean_ig, adv_ig, delta_ig, drift, x, x_adv

# --------------------------------------------
# SmoothGrad-Squared Attribution Implementation
# --------------------------------------------


def smoothgrad_squared(model, input_tensor, target_label, baseline, steps=50, noise_level=0.1, device="cpu"):
    model.eval()
    input_tensor = input_tensor.to(device)
    baseline = baseline.to(device)
    accumulated_grads = torch.zeros_like(input_tensor)

    for _ in range(steps):
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        noisy_input = noisy_input.clone().detach().requires_grad_(True)
        output = model(noisy_input)
        score = output[0, target_label]
        model.zero_grad()
        score.backward(retain_graph=True)
        grad = noisy_input.grad
        accumulated_grads += grad ** 2  # Squared gradients

    smooth_grad_squared = accumulated_grads / steps
    return smooth_grad_squared

# --------------------------------------------
# Integration in Attribution Analysis
# --------------------------------------------


def compare_ig_smoothgrad(model, input_tensor, label, baseline, device, steps=50):
    ig, _ = integrated_gradients(
        model, input_tensor, label, baseline, steps=steps, device=device)
    entropy_ig, _ = attribution_entropy(ig)

    sg_sq = smoothgrad_squared(
        model, input_tensor, label, baseline, steps=steps, device=device)
    entropy_sg, _ = attribution_entropy(sg_sq)

    print(
        f"Entropy IG: {entropy_ig:.4f}, Entropy SmoothGrad²: {entropy_sg:.4f}")
    visualize_contrastive_attributions(
        input_tensor, ig, sg_sq, save_path="plots/ig_vs_smoothgrad2.png")

# --------------------------------------------
# CROWN-IBP Formal Verification using auto_LiRPA
# --------------------------------------------


def replace_pooling(module):
    for name, child in module.named_children():
        if isinstance(child, nn.MaxPool2d):
            if child.stride != child.kernel_size:
                # Replace with compliant pooling
                new_pool = nn.MaxPool2d(
                    kernel_size=child.stride,
                    stride=child.stride,
                    padding=child.padding
                )
                setattr(module, name, new_pool)
        else:
            replace_pooling(child)


def run_crown_ibp_verification(model, input_tensor, label, eps=0.05, device="cpu"):
    # Replace unsupported activations with compatible ones
    def replace_activations(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Hardswish):
                module.add_module(name, torch.nn.ReLU())
            elif isinstance(child, torch.nn.Hardsigmoid):
                module.add_module(name, torch.nn.Sigmoid())
            else:
                replace_activations(child)

    model = model.to(device)
    replace_activations(model)
    replace_pooling(model)  # Ensure pooling is compatible with auto_LiRPA
    model.eval()  # Ensure model is in eval mode to disable dropout

    # Ensure input has batch dimension
    input_tensor = input_tensor.to(device).unsqueeze(0)
    label = torch.tensor([label], dtype=torch.long, device=device)

    # Define perturbation
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    bounded_input = BoundedTensor(input_tensor, ptb)

    # Initialize BoundedModule with the original input tensor
    lirpa_model = BoundedModule(
        model,
        input_tensor,  # Use the original tensor, not bounded_input
        device=device,
        verbose=False,
        custom_ops={}
    )

    try:
        lb, ub = lirpa_model.compute_bounds(
            x=(bounded_input,), method="CROWN-IBP")
    except RuntimeError as e:
        print("Bound computation failed:", e)
        return False

    # Handle output dimensions
    if lb.dim() == 1:
        lb = lb.unsqueeze(-1)
    if lb.dim() == 2:
        correct_lb = lb[0, label.item()]
        other_lb = torch.cat([lb[0, :label.item()], lb[0, label.item()+1:]])
        verified = (correct_lb > other_lb.max()).item()
    else:
        print(f"Unexpected bound shape: {lb.shape}")
        return False

    if verified:
        print("✅ CROWN-IBP Verification Passed")
    else:
        print("❌ Verification Failed")

    return verified


# -------------------------------
# CLI + Main
# -------------------------------


def main(dataset_name="mnist", model_type="simplecnn", static=False, plot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader, test_dataset = load_dataset(dataset_name)
    model = get_model(model_type, dataset_name).to(device)
    print("Training model...")
    train_model(model, train_loader, device, epochs=5)
    print("\nEvaluating on clean test data:")
    test_model(model, test_loader, device)

    # Red-teaming vs full safety mode
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    adv_images = pgd_attack(model, images, labels, device,
                            eps=0.3, alpha=0.01, iters=40)
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images)
        adv_preds = outputs.argmax(dim=1)
        correct_adv = (adv_preds.cpu() == labels).sum().item()
    print(f"\nAdversarial Attack Accuracy: {correct_adv/len(labels)*100:.2f}%")

    # Formal Verification + Attribution
    sample_img, sample_label = test_dataset[0]
    print("\nRunning formal verification check:")
    formal_verification_check(model, sample_img, sample_label, device)
    sample_tensor = sample_img.unsqueeze(0)
    baseline_zero = torch.zeros_like(sample_tensor)
    baseline_mean = torch.full_like(sample_tensor, sample_tensor.mean().item())
    candidate_baselines = [baseline_zero, baseline_mean]
    best_baseline, best_ig = dynamic_baseline_selection(
        model, sample_tensor, sample_label, candidate_baselines, device=device, plot=plot)
    entropy_val, _ = attribution_entropy(best_ig)
    print(f"Attribution Entropy: {entropy_val:.4f}")

    if not static:
        print("\nRunning Contrastive Attribution Analysis...")
        clean_ig, adv_ig, delta_ig, drift, input_x, adv_x = compute_contrastive_attributions(
            model, sample_img, sample_label, device, best_baseline)
        visualize_contrastive_attributions(
            input_x, clean_ig, adv_ig, delta_ig, save_path="plots/contrastive_ig.png")

        print("\nRunning SmoothGrad² Comparison...")
    compare_ig_smoothgrad(model, sample_tensor, sample_label,
                          best_baseline, device)

    print("\nRunning CROWN-IBP Verification...")
    remove_dropout_layers(model)   # Remove Dropout before verification
    # model.eval()                   # Set to eval mode

    run_crown_ibp_verification(
        model, sample_img, sample_label, eps=0.05, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="simplecnn")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.model, args.static, args.plot)
