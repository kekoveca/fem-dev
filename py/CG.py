import numpy as np


def CG(A, b, rtol=1e-8, atol=1e-12, m=128):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]

    b_norm = np.linalg.norm(b)

    if m is None:
        m = n

    x_curr = np.zeros_like(b, dtype=float)
    r_curr = b - A @ x_curr
    beta = np.linalg.norm(r_curr)

    if beta < atol:
        return x_curr.copy(), np.array([beta])

    p_curr = r_curr
    r_next = np.zeros_like(r_curr)
    x_next = np.zeros_like(x_curr)

    residuals = np.empty(m, dtype=float)

    for j in range(m):
        a = np.dot(r_curr, r_curr) / np.dot((A @ p_curr), p_curr)
        x_next = x_curr + a * p_curr
        r_next = r_curr - a * (A @ p_curr)
        beta = np.dot(r_next, r_next) / np.dot(r_curr, r_curr)
        p_next = r_next + beta * p_curr

        residual = np.linalg.norm(r_next)
        residuals[j] = residual

        if residual < atol + rtol * b_norm:
            return x_next, residuals[: j + 1]

        x_curr = x_next
        r_curr = r_next
        p_curr = p_next

    raise RuntimeError("CG did not converge")
