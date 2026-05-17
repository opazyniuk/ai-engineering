"""
Крок 1.2: збираємо train/val loss у списки під час тренування,
будуємо matplotlib графік з відміткою точки мінімального val loss.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import nano_gpt as ng


def main():
    print(f"device={ng.device} vocab_size={ng.vocab_size} chars={len(ng.TEXT):,}")
    model = ng.NanoGPT().to(ng.device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=ng.LR)

    iters, train_losses, val_losses = [], [], []

    t_start = time.perf_counter()
    for it in range(ng.MAX_ITERS + 1):
        if it % ng.EVAL_INTERVAL == 0:
            l = ng.estimate_loss(model)
            iters.append(it)
            train_losses.append(l["train"])
            val_losses.append(l["val"])
            elapsed = time.perf_counter() - t_start
            print(f"iter {it:4d} | train {l['train']:.3f} | val {l['val']:.3f} | {elapsed:.0f}s")

        xb, yb = ng.get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"\ntraining done in {time.perf_counter() - t_start:.0f}s")

    best_idx = int(np.argmin(val_losses))
    best_iter = iters[best_idx]
    best_val = val_losses[best_idx]
    best_train = train_losses[best_idx]
    final_gap = val_losses[-1] - train_losses[-1]
    print(f"\nmin val loss = {best_val:.3f} at iter {best_iter}")
    print(f"train at that iter = {best_train:.3f}")
    print(f"final train/val gap = {final_gap:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, train_losses, marker="o", label="train")
    ax.plot(iters, val_losses, marker="s", label="val")
    ax.axvline(
        best_iter,
        color="red",
        linestyle="--",
        label=f"min val = {best_val:.3f} @ iter {best_iter}",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(
        f"nano-GPT training on Ukrainian transcripts ({len(ng.TEXT):,} chars, vocab={ng.vocab_size})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = "plots/loss.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nplot saved to {out_path}")


if __name__ == "__main__":
    main()