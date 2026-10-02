# vendored: Stability-AI/stable-audio-tools 3241adba4fc2a85cf5b29d9eb68d42f40a28e820 (MIT)

import sys

from . import stable_audio_tools as stable_audio_tools_vendor

sys.modules["stable_audio_tools"] = stable_audio_tools_vendor
