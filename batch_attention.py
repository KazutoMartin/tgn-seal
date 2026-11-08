"""
Times one attention call (forward + backward) for each kernel, at the
shapes your Transformer link predictor sees. Run on one A6000:

    python bench_attention.py --steps_per_epoch <"num of batches per epoch" from your log>

Defaults match your code: 200 subgraphs per call (batch size 200), 2 calls per
training step (positive and negative loaders), 2 Transformer layers, 2 heads,
width 172. --nodes is the padded subgraph size; 222 is the upper bound for
k=2 and n=10 (2 + 2*10 + 2*100), so it is the worst case.

Compare the "per epoch" column with your measured epoch time (e.g. 132.9 s on
Dept 1) to see how much of an epoch attention can possibly account for.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

p = argparse.ArgumentParser()
p.add_argument("--subgraphs", type=int, default=200)
p.add_argument("--calls_per_step", type=int, default=2)
p.add_argument("--nodes", type=int, default=222)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--steps_per_epoch", type=int, default=215, help="training batches per epoch")
p.add_argument("--iters", type=int, default=200)
a = p.parse_args()

dev = "cuda"
B, N = a.subgraphs, a.nodes

# (label, width, heads, dtype, use padding mask, backend)
configs = [
    ("CURRENT  fp32  d=172 (head 86)  mask  -> math", 172, 2, torch.float32, True, SDPBackend.MATH),
    ("fp32  d=176 (head 88)  mask  -> mem-efficient", 176, 2, torch.float32, True, SDPBackend.EFFICIENT_ATTENTION),
    ("bf16  d=176 (head 88)  mask  -> mem-efficient", 176, 2, torch.bfloat16, True, SDPBackend.EFFICIENT_ATTENTION),
    ("bf16  d=176 (head 88)  mask  -> math", 176, 2, torch.bfloat16, True, SDPBackend.MATH),
    ("bf16  d=176 (head 88)  NO mask -> flash", 176, 2, torch.bfloat16, False, SDPBackend.FLASH_ATTENTION),
]

print(f"{B} subgraphs x {N} padded nodes, {a.layers} layer(s), {a.calls_per_step} calls/step, {a.steps_per_epoch} steps/epoch")
print("Times cover the attention core (forward + backward) only.\n")
print(f"{'configuration':50s} {'ms/call':>9s} {'s/epoch':>9s}")
for label, d, h, dt, use_mask, backend in configs:
    q = torch.randn(B, h, N, d // h, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = None
    if use_mask:
        lengths = torch.randint(N // 4, N + 1, (B,), device=dev)  # realistic varying subgraph sizes
        mask = (torch.arange(N, device=dev)[None, :] < lengths[:, None])[:, None, None, :]
    try:
        with sdpa_kernel(backend):
            for _ in range(20):  # warm-up
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask).sum().backward()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.iters):
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask).sum().backward()
            end.record()
            torch.cuda.synchronize()
        ms = start.elapsed_time(end) / a.iters
        print(f"{label:50s} {ms:9.3f} {ms * a.layers * a.calls_per_step * a.steps_per_epoch / 1000:9.2f}")
    except RuntimeError:
        print(f"{label:50s} {'not available':>19s}")