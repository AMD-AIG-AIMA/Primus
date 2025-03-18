###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


def call_once(func):
    called = False

    def wrapper(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return func(*args, **kwargs)
        else:
            assert True, f"{func.__name__} can only be called once!"

    return wrapper
