import time


class _Timestamp:
    @property
    def now(self) -> str:
        return time.strftime("%H:%M:%S")


Timestamp = _Timestamp()
