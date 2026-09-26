import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bad = 0
for k in a.files:
    x, y = a[k], b[k]
    n = int((x.view(np.uint32) != y.view(np.uint32)).sum())
    if n:
        bad += 1
        d = np.abs(x - y); print(k, 'mismatch', n, 'maxabs', float(d.max()), 'maxrel', float((d / (np.abs(x) + 1e-30)).max()))
print('compared', len(a.files), 'arrays;', 'ALL IDENTICAL' if not bad else f'{bad} differ')
