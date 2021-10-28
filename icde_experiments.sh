#!/bin/bash

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin icde_vs_deletion_performance \
  > results__icde_vs_deletion_performance.txt