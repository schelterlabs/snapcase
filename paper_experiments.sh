#!/bin/bash

#RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_tifu_deletion_performance -- \
#  > results__paper_tifu_deletion_performance.txt

#RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_vs_deletion_performance -- \
#  > results__paper_vs_deletion_performance.txt

#RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_tifu_incremental_performance -- \
#  > results__paper_tifu_incremental_performance.txt

#RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_vs_incremental_performance -- \
#  > results__paper_vs_incremental_performance.txt

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_vs_mixed_performance -- \
  > results__paper_vs_mixed_performance.txt

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_tifu_mixed_performance -- \
  > results__paper_tifu_mixed_performance.txt