import cvxpy as cx
import numpy as np
import torch
import torch.nn.functional as F


def tune_temp(
    logits,
    labels,
    binary_search=True,
    lower=0.2,
    upper=5.0,
    eps=0.0001,
):
    logits = np.array(logits)

    if binary_search:

        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        t_guess = torch.FloatTensor([0.5 * (lower + upper)]).requires_grad_()

        while upper - lower > eps:
            if (
                torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[
                    0
                ]
                > 0
            ):
                upper = 0.5 * (lower + upper)
            else:
                lower = 0.5 * (lower + upper)
            t_guess = t_guess * 0 + 0.5 * (lower + upper)

        t = min(
            [lower, 0.5 * (lower + upper), upper],
            key=lambda x: float(F.cross_entropy(logits / x, labels)),
        )
    else:

        set_size = np.array(logits).shape[0]

        t = cx.Variable()

        expr = sum(
            (
                cx.Minimize(cx.log_sum_exp(logits[i, :] * t) - logits[i, labels[i]] * t)
                for i in range(set_size)
            )
        )
        p = cx.Problem(expr, [lower <= t, t <= upper])

        p.solve()  # p.solve(solver=cx.SCS)
        t = 1 / t.value

    return t
