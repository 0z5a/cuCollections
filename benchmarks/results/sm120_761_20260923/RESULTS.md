# cuCollections #761 — RTX 5090 continuation (2026-09-23)

## Result

`blocked_reproduction`: current cuCO `dev` builds and its standalone `static_multiset` insert/count/retrieve path passes on SM120. The historical [mixed-join issue](https://github.com/NVIDIA/cuCollections/issues/761) requires the cuDF consumer path; this 0z5a environment has no preinstalled cuDF library, and the installed CMake 3.22.1 cannot configure current cuCO's RAPIDS CMake 4 requirement. No cuDF mixed-join E2E, current spill regression, or production speedup is claimed. The issue and [follow-up #847](https://github.com/NVIDIA/cuCollections/issues/847) are closed.

## Provenance and checks

- Original 8× NVIDIA GeForce RTX 5090 host, GPU 4; driver 580.82.07; CUDA toolkit 13.0.88. Work was confined to `/home/gongji/0z5a-work` on local storage.
- cuCO `dev` SHA `0104ea65d9e894fde2598f14c9148ab5280151cd`; clean CCCL header snapshot `d6842f00dd44f434ecc761c81a4efbfd9a6249c8`; direct `nvcc -O3 -std=c++17 -arch=sm_120 --expt-extended-lambda -lineinfo` build. No installed package was changed.
- `examples/static_multiset/host_bulk_example.cu` ran and reported `Success! Found all keys.` Its count kernel used 36 registers, retrieve 40, insert 32; `ptxas` reported zero spill loads/stores for each. This is a cuCO example result, not a cuDF mixed-join profile. [Build log](host_bulk_build.log).
- An isolated [capacity sensitivity probe](capacity_count_probe.cu) inserted the same 1,048,576 unique integer keys into two sets, then counted all hits or all misses. Actual capacities were 2,097,176 and 4,194,472 slots. Each count includes the host API call and synchronization. Four warmups preceded 24 alternating-order pairs; all counts matched expected results. [Raw paired timings](capacity_count_gpu4.csv), [build log](capacity_build.log). Binary SHA256: `6df4c14cb291fa9bf97ca67a5bcf266b72cf7e032d476c8a18643439c8749d10`.

| Probe | Smaller capacity median (µs) | Larger capacity median (µs) | Smaller-capacity speedup, paired median | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| All hits | 39.166 | 38.938 | 0.989× | 0.977–1.020× |
| All misses | 38.838 | 39.169 | 1.013× | 0.972–1.039× |

Speedup is larger-capacity latency divided by smaller-capacity latency. Intervals resample the 24 pairs with a fixed seed and describe within-run variability only. Neither probe shows a stable difference. This only tests a simple cuCO count workload; the historical report explicitly states that its mixed-join slowdown did not reproduce in standalone cuCO benchmarks.

## Gate for further work

Run the current cuDF mixed-join benchmark and correctness fixture in an existing compatible 0z5a environment, record its pinned cuCO SHA and output count, then compare a minimal candidate only if the current regression and spill traffic reproduce. No optimization candidate was created from this standalone result.
