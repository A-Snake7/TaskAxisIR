BLIND_MODES = ("blind_soft", "blind_v2", "blind_v3")

# These modes train from a local restoration crop plus a fixed-size view of
# the complete degraded image. Keeping this distinction in one place avoids
# silently dropping the global context in a new evaluation entry point.
MULTIVIEW_BLIND_MODES = ("blind_v2", "blind_v3")


def is_blind_mode(condition_mode):
    return condition_mode in BLIND_MODES


def is_multiview_blind_mode(condition_mode):
    return condition_mode in MULTIVIEW_BLIND_MODES
