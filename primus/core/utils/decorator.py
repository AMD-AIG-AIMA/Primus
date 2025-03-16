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
