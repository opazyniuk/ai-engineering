"""
Крок 1.4 (частина 1): тренування + збереження чекпоінта.
checkpoint.pt містить state_dict, config (для відтворення архітектури) і vocab (токенайзер).
"""
import time
import torch
import nano_gpt as ng


CKPT_PATH = "checkpoint.pt"


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
    print(f"\ntraining done in {time.perf_counter() - t_start:.0f}s")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "BLOCK_SIZE": ng.BLOCK_SIZE,
            "N_EMBED": ng.N_EMBED,
            "N_HEAD": ng.N_HEAD,
            "N_LAYER": ng.N_LAYER,
            "DROPOUT": ng.DROPOUT,
            "vocab_size": ng.vocab_size,
        },
        "vocab": {
            "stoi": ng.stoi,
            "itos": ng.itos,
        },
    }
    torch.save(ckpt, CKPT_PATH)
    import os
    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"checkpoint saved to {CKPT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
