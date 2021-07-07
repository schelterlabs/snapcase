#!/usr/bin/env fish

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w1 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-1-100-20-100.txt 100 20 100 0

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w2 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-2-100-20-100.txt 100 20 100 0

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w3 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-3-100-20-100.txt 100 20 100 0


RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w1 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-1-100-20-100-5.txt 100 20 100 5

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w2 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-2-100-20-100-5.txt 100 20 100 5

RUSTFLAGS="-C target-cpu=native" cargo run --release --bin vsknn_experiments -- \
-w3 datasets/session-based/bolcom-clicks-1m_train.txt datasets/session-based/bolcom-clicks-1m_test.txt \
l-3-100-20-100-5.txt 100 20 100 5
