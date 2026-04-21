from types import SimpleNamespace


class _ConfigProxy:
    def __init__(self):
        self._config = None

    def set(self, cfg):
        self._config = cfg

    def get(self):
        if self._config is None:
            raise RuntimeError("Active config is not initialized.")
        return self._config

    def __getattr__(self, item):
        return getattr(self.get(), item)


config = _ConfigProxy()


def set_active_config(cfg: SimpleNamespace):
    config.set(cfg)


def get_active_config() -> SimpleNamespace:
    return config.get()
