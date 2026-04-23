import numpy as np


def GMRES(A, b, rtol=1e-8, atol=1e-12, m=128):
    b = np.asarray(b, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)

    b_norm = np.linalg.norm(b)

    n = b.shape[0]

    if m is None:
        m = n

    x0 = np.zeros_like(b, dtype=float)
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)

    if beta < atol:
        return x0.copy()

    V = np.zeros((n, m + 1), dtype=float)
    H = np.zeros((m + 1, m), dtype=float)

    g = np.zeros(m + 1, dtype=float)
    g[0] = beta

    cs = np.zeros(m, dtype=float)
    sn = np.zeros(m, dtype=float)

    V[:, 0] = r0 / beta

    residuals = np.zeros(m)
    residuals.fill(np.nan)

    for j in range(m):
        wj = A @ V[:, j]

        for i in range(j + 1):
            H[i, j] = np.dot(wj, V[:, i])
            wj -= H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(wj)
        happy_breakdown = H[j + 1, j] < atol

        if not happy_breakdown:
            V[:, j + 1] = wj / H[j + 1, j]

        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        rho = np.hypot(H[j, j], H[j + 1, j])

        if rho == 0.0:
            cs[j] = 1.0
            sn[j] = 0.0
        else:
            cs[j] = H[j, j] / rho
            sn[j] = H[j + 1, j] / rho

        temp = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j, j] = temp
        H[j + 1, j] = 0.0

        temp = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
        g[j] = temp

        residual = abs(g[j + 1])
        residuals[j] = residual

        if residual / b_norm < rtol or happy_breakdown:
            ym = np.linalg.solve(H[: j + 1, : j + 1], g[: j + 1])
            xm = x0 + V[:, : j + 1] @ ym
            return xm, residuals

    raise RuntimeError("GMRES did not converge")
