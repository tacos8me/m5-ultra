import torch
torch.manual_seed(0)
for D in (128, 512):
    w = (1 + 0.1 * torch.randn(D, device="cuda")).float()
    x = torch.randn(8192, D, device="cuda").to(torch.bfloat16) * 3
    f = lambda t: (w * (t.float() * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16)
    ref = f(x)
    bad = [m for m in list(range(1, 70)) + [100, 1000] if not torch.equal(f(x[:m].contiguous()), ref[:m])]
    # also inside a CUDA graph at small M
    xs = x[:4].clone(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = f(xs)
    g.replay(); torch.cuda.synchronize()
    print({"D": D, "torch_rows_differ_at_M": bad, "graph_equal": torch.equal(out, ref[:4])})
