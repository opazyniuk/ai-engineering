"""
Крок 1.1: тренуємо baseline на власному датасеті, генеруємо 3 семпли.
nano_gpt.py не модифікується — імпортуємо як модуль.
"""
import time
import torch
import nano_gpt as ng


def main():
    print(f"device={ng.device} vocab_size={ng.vocab_size} chars={len(ng.TEXT):,}")
    model = ng.NanoGPT().to(ng.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=ng.LR)

    t_start = time.perf_counter()
    for it in range(ng.MAX_ITERS + 1):
        if it % ng.EVAL_INTERVAL == 0:
            l = ng.estimate_loss(model)
            elapsed = time.perf_counter() - t_start
            print(f"iter {it:4d} | train {l['train']:.3f} | val {l['val']:.3f} | {elapsed:.0f}s")

        xb, yb = ng.get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"\ntraining done in {time.perf_counter() - t_start:.0f}s")

    for i, seed in enumerate([1337, 42, 7], start=1):
        torch.manual_seed(seed)
        ctx = torch.zeros((1, 1), dtype=torch.long, device=ng.device)
        out = model.generate(ctx, max_new_tokens=300, temperature=1.0)[0].tolist()
        print(f"\n=== SAMPLE {i} (seed={seed}) ===")
        print(ng.decode(out))


if __name__ == "__main__":
    main()