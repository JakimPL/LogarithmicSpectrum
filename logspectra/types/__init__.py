from typing import Literal, TypeAlias, Union

import numpy as np

Float: TypeAlias = Union[float, np.floating]
SampleRate: TypeAlias = Literal[16000, 22050, 32000, 44100, 48000, 96000, 192000]
FFTSize: TypeAlias = Literal[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
