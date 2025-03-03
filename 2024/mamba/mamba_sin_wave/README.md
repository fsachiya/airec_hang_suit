version3

pb: 1*10
linear: 1*1 -> 1*1
norm: min_range - init_fast_tau ~ max_range - init_fast_tau
fast_tau = fast_tau + norm(sigmoid(linear(pb)))