"""
Крок 1.3: temperature × top_k таблиця 3×3.
Тренує модель один раз, генерує 9 семплів з різними комбінаціями,
зберігає у таблицю в логу.
"""
import time
import torch
import nano_gpt as ng


TEMPERATURES = [0.5, 1.0, 1.5]
TOP_KS = [5, 20, None]      # None = "all", без обмеження
SAMPLE_LEN = 250            # коротше за 300, щоб таблиця читалась
SEED = 1337                 # один сід для всіх — щоб різниця була ТІЛЬКИ через temp/top_k


def main():
    print(f"device={ng.device} vocab_size={ng.vocab_size} chars={len(ng.TEXT):,}")
    model = ng.NanoGPT().to(ng.device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

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
    print(f"\ntraining done in {time.perf_counter() - t_start:.0f}s\n")

    print("=" * 80)
    print(f"3x3 SAMPLES — fixed seed={SEED}, length={SAMPLE_LEN}")
    print("=" * 80)
    for temp in TEMPERATURES:
        for top_k in TOP_KS:
            torch.manual_seed(SEED)
            ctx = torch.zeros((1, 1), dtype=torch.long, device=ng.device)
            out = model.generate(
                ctx, max_new_tokens=SAMPLE_LEN, temperature=temp, top_k=top_k
            )[0].tolist()
            label_k = top_k if top_k is not None else "None"
            header = f"--- temp={temp} | top_k={label_k} ---"
            print(f"\n{header}")
            print(ng.decode(out))


if __name__ == "__main__":
    main()
