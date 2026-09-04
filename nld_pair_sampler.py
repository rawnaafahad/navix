"""
nld_pair_sampler.py — wraps TtyrecDataset to yield (past_frame, future_frame)
pairs at FUTURE_HORIZON apart. Deliberately drops keypresses from what the
discovery model sees.
"""

import nle.dataset as nld
import numpy as np

FUTURE_HORIZON = 16

def make_pair_batches(dataset_name, dbfilename, batch_size, seq_length=64):
    dataset = nld.TtyrecDataset(
        dataset_name, batch_size=batch_size, seq_length=seq_length, dbfilename=dbfilename
    )
    for mb in dataset:
        chars = mb["tty_chars"]  # (B, T, 24, 80) -- keypresses INTENTIONALLY unused here
        B, T = chars.shape[0], chars.shape[1]
        if T <= FUTURE_HORIZON:
            continue
        # sample one (t, t+horizon) pair per sequence in this minibatch
        t = np.random.randint(0, T - FUTURE_HORIZON, size=B)
        past = chars[np.arange(B), t]
        future = chars[np.arange(B), t + FUTURE_HORIZON]
        yield past, future
