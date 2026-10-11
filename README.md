# Chudnovsky π (multi-core)

Compute millions of digits of π in seconds using the **Chudnovsky series**, **binary splitting**,
and a multi-core process pool, with [`gmpy2`](https://gmpy2.readthedocs.io/) doing the big-number arithmetic.

The project began with a post about the slowly converging Wallis product, which led to Ramanujan's
series for 1/π and the Chudnovsky formula, which gives about 14.18 correct digits per term.

## Contents

| File | Purpose |
|---|---|
| `Digits_of_pi_Chudnovksy_multi_core.ipynb` | The notebook: calculate, save and verify digits; Wallis vs. Chudnovsky convergence; time-distribution benchmark |
| `pi_worker.py` | Binary splitting (`bs_step`), the pairwise tree merge, and `calculate_pi_parallel()` with timing checkpoints |

`pi_worker.py` must sit next to the notebook. It is a separate module because macOS starts worker
processes with `spawn`, which cannot see functions defined in a notebook cell.

## What the notebook does

1. **Calculate, save and verify.** Prompts for the number of digits, computes π across all CPU cores,
   saves `pi_<N>_digits.txt` (`3.` followed by the decimals, 80 characters per line), and checks the
   result against `gmpy2.const_pi()` and, independently, against the
   [Pi Delivery API](https://pi.delivery/) (every digit up to 200,000; a random sample of chunks above that).
   Any error is reported with its position, the calculated digit and the true digit.
2. **Wallis vs. Chudnovsky.** Plots correct digits against the number of terms for each method.
3. **Time distribution vs. digit scale.** Stacked bar and area charts showing how the time splits between
   pool startup, parallel compute, merge, `sqrt` + divide, and string formatting.

## Method

With `P`, `Q`, `T` computed by binary splitting over a range of terms,

```
pi = 426880 * sqrt(10005) * Q / T
```

The `13591409` constant is already folded into `T` through term 0, so the denominator is just `T`.
The terms are split into one chunk per core, each chunk is computed in its own process, and the partial
results are merged pairwise as a balanced tree. The tree merge was about 3.5x faster than a left-to-right
chain for the merge step, and up to about 2.6x faster for the whole pipeline at 30 million digits.

## Results (32-core Apple M3 Ultra)

| Digits | Time |
|---|---|
| 3,000,000 | about 1.5 s (before the tree merge) |
| 30,000,000 | about 8.4 s for calculation and merge (with the tree merge) |

Timings are not hardware-counter measurements, so they cannot show on their own whether a run is
memory-bandwidth-bound. Memory use was about 130 bytes per digit at 30 million digits.

## Requirements

- Python 3
- `gmpy2`, `numpy`, `matplotlib`

```bash
pip install gmpy2 numpy matplotlib
```

Then open the notebook in Jupyter or VS Code and run the cells in order. If you edit `pi_worker.py`,
restart the kernel so the worker processes pick up the change.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
