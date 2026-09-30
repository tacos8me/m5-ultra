"""pytest plugin (ds41 CPU test runs, og_serve/cpu_suite.sh): MLX default device pinned to the CPU, Metal kernels refuse to run."""


class _Blocked:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'GPU blocked (ds41 CPU run): metal_kernel {self.name}')


def pytest_configure(config):
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001
        return
    mx.set_default_device(mx.cpu)
    real_set = mx.set_default_device

    def pinned(device):
        if 'gpu' in str(device).lower():
            return real_set(mx.cpu)
        return real_set(device)

    mx.set_default_device = pinned
    mx.fast.metal_kernel = lambda *a, **k: _Blocked(k.get('name', a[0] if a else '?'))
