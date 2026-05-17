"""
Крок 2.2: візуалізація attention weights (heatmap).

Завантажує checkpoint, прокачує короткий input через модель,
дістає att-матрицю з кожного шару/голови (через self.last_attn),
малює grid heatmap-ів 4 layers × 6 heads.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

import nano_gpt as ng


CKPT_PATH = "checkpoint.pt"
PROMPT = "Президент України Володимир"
OUT_PATH = "plots/attention_heatmap.png"


def main() -> None:
    print(f"loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=ng.device, weights_only=False)
    stoi = ckpt["vocab"]["stoi"]

    model = ng.NanoGPT().to(ng.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params on {ng.device}")

    ids = [stoi[c] for c in PROMPT if c in stoi]
    x = torch.tensor([ids], dtype=torch.long, device=ng.device)
    print(f"prompt={PROMPT!r}  len={len(ids)} tokens")

    with torch.no_grad():
        _ = model(x)

    n_layers = ng.N_LAYER
    n_heads = ng.N_HEAD
    print(f"capturing attention: {n_layers} layers × {n_heads} heads = {n_layers*n_heads} maps")

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * 2.4, n_layers * 2.4),
        squeeze=False,
    )
    fig.suptitle(
        f"Attention heatmaps — prompt: {PROMPT!r} ({len(ids)} tokens)\n"
        f"rows: query position | cols: key position | brighter = more attention",
        fontsize=11,
    )

    tick_labels = list(PROMPT)

    for layer_idx, block in enumerate(model.blocks):
        att = block.attn.last_attn          # (1, n_heads, T, T)
        att = att[0].cpu().numpy()          # (n_heads, T, T)

        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            ax.imshow(att[head_idx], cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=9)

            if layer_idx == n_layers - 1:
                ax.set_xticks(range(len(tick_labels)))
                ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
            else:
                ax.set_xticks([])

            if head_idx == 0:
                ax.set_yticks(range(len(tick_labels)))
                ax.set_yticklabels(tick_labels, fontsize=6)
            else:
                ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")

    print("\nstats per layer (entropy of average attention distribution):")
    for layer_idx, block in enumerate(model.blocks):
        att = block.attn.last_attn[0]  # (n_heads, T, T)
        eps = 1e-9
        ent = -(att * (att + eps).log()).sum(dim=-1)  # (n_heads, T)
        mean_ent = ent.mean().item()
        max_ent = np.log(len(ids))
        print(f"  layer {layer_idx}: mean entropy = {mean_ent:.3f} (max possible = {max_ent:.3f})")


if __name__ == "__main__":
    main()
