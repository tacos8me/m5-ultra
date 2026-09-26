import torch
x = torch.empty(1500 * 2**20, dtype=torch.uint8, device='cuda').random_(0, 255)
f = x.view(torch.float32)
h = x.view(torch.bfloat16)
for name, fn in [('sum fp32', lambda: f.sum()), ('amax bf16', lambda: h.amax()), ('sum int64 view', lambda: x.view(torch.int64).sum())]:
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); e1.synchronize()
        best = min(best, e0.elapsed_time(e1) / 1e3)
    print(f'{name}: {x.numel() / best / 1e12:.3f} TB/s')
