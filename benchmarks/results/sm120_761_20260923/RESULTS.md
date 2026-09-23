# cuCollections #761 — RTX 5090 continuation (2026-09-23)

## Result

`consumer_e2e_pass_without_reproduced_regression`: current cuCO `dev` builds and its standalone `static_multiset` insert/count/retrieve path passes on SM120. A separate isolated `0z5a-ptx/cudf` environment now runs the [pylibcudf mixed-join consumer API](mixed_join_e2e.py) end to end on the same RTX 5090. Both inner and semi joins return the expected rows. The installed cuDF 26.8.1 wheel is precompiled, so its embedded cuCO revision is not established by this test and its timings are not a comparison against the fixed cuCO `dev` SHA. No current spill regression or production speedup is claimed. The historical [issue #761](https://github.com/NVIDIA/cuCollections/issues/761) and [follow-up #847](https://github.com/NVIDIA/cuCollections/issues/847) are closed.

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

## cuDF mixed-join consumer E2E

- New isolated `/home/gongji/0z5a-work/0z5a-ptx/cudf` environment; `cudf-cu13`, `pylibcudf-cu13`, and `libcudf-cu13` 26.8.1 from PyPI wheels. All 33 transferred dependency wheels passed [SHA256 verification](cudf-wheel-sha256.txt). The original `/home/gongji/0z5a` environment was unchanged.
- [Harness](mixed_join_e2e.py) checks a mixed equality-key plus `left.value < right.value` predicate: five expected inner-join pairs and four expected semi-join rows. Both passed on GPU 4. [Raw log](mixed_join_e2e.log).
- Inputs below contain unique int32 keys on each side in shuffled order and random int32 predicate values. Each timing includes the pylibcudf API call, hash-table construction, output allocation, and GPU synchronization; Arrow conversion is outside the timed region. Two warmups preceded ten repetitions.

| Consumer operation | Rows per side | Median (ms) | Min–max (ms) | Compared speedup |
| --- | ---: | ---: | ---: | ---: |
| `mixed_inner_join` | 1,048,576 | 0.961 | 0.939–0.991 | N/A — no candidate |
| `mixed_left_semi_join` | 1,048,576 | 0.580 | 0.572–0.605 | N/A — no candidate |
| `mixed_inner_join` | 4,194,304 | 1.787 | 1.775–1.821 | N/A — no candidate |
| `mixed_left_semi_join` | 4,194,304 | 1.744 | 1.727–1.755 | N/A — no candidate |

The consumer run verifies actual mixed-join execution and correctness on SM120. Since the cuDF wheel's cuCO SHA is unknown, it does not prove that the standalone cuCO snapshot has the same production latency. No optimization candidate was created from the capacity probe's 0.989–1.013× paired medians.
