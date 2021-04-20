# rapidAligner

## About

rapidAligner is a CUDA-accelerated library for the windowed alignment of time series implemented in Numba and Cupy. The library supports the following alignment modes: windowed alignment of a query in a stream of time series data

- sdist: using plain Euclidean distance (L2 norm-induced metric)
- mdist: using locally mean-adjusted Euclidean distance
- zdist: using locally mean- and amplitude-adjusted Euclidean distance (isomorphic to windowed Pearson's correlation coefficient)

## Compute Modes

rapidAligner supports 2 compute modes for all aforementioned distance measures to trade-off memory versus speed. Assume a query of length m and a stream of length n:

- naive: all n-m+1 windowed alignment candidates are normalized and compared individually with a minimal memory footprint but O(n*m) asymptotic computational complexity. This mode is still reasonable fast using warp-aggregated statistics and accumulation schemes.
- FFT: If m > log_2(n), we can exploit the Convolution Theorem to accelerate the computation significantly resulting in O(n log n) runtime but a higher memory footprint. This compute mode is fully independent of the query's length and thus advisable for large input. The higher memory usage is mainly caused by computationally fast but memory-intensive (out-of-place) primitives such as CUDA-accelerated Fast Fourier Transforms and Prefix Scans.

## Usage

See the notebooks folder for examplary usage.
