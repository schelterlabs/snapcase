#!/bin/bash

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin paper_tifu_deletion_performance -- \
  > results__paper_tifu_deletion_performance.txt

#RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vldb_vs_deletion_performance -- \
#  > results__vldb_vs_deletion_performance.txt
