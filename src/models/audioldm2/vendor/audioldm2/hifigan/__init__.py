# vendored: haoheliu/audioldm2 b5786c5dc0ae8f766337fdc1b67ab6046586d14d (CC-BY-NC-SA-4.0)
from .models_v2 import Generator
from .models import Generator as Generator_old


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
