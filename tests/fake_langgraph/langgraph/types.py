_NO_RESUME = object()
_current = [_NO_RESUME]


def _set_resume(v):
    _current[0] = v


class GraphInterrupt(Exception):
    def __init__(self, value):
        self.value = value


class Interrupt:
    def __init__(self, value):
        self.value = value


class Command:
    def __init__(self, resume=None):
        self.resume = resume


def interrupt(value):
    if _current[0] is not _NO_RESUME:
        v, _current[0] = _current[0], _NO_RESUME
        return v
    raise GraphInterrupt(value)
