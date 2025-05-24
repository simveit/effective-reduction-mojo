Execute `make run all` to run compile and run all the kernels.

`H100, N = 2^30`
| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| two_pass_1 | 622.87 | 18.87% | Two-pass |
| two_pass_2 | 718.54 | 21.77% | Two-pass |
| two_pass_3 | 633.19 | 19.19% | Two-pass |
| two_pass_4 | 738.78 | 22.39% | Two-pass |
| one_pass_1 | 985.32 | 29.86% | One-pass |
| one_pass_2 | 1881.55 | 57.02% | One-pass |
| one_pass_3 | 1882.55 | 57.05% | One-pass |
| one_pass_4 | 3134.26 | 94.98% | One-pass |
| one_pass_5 | 2547.15 | 77.19% | One-pass |