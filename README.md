# LibRVC
An easy-to-use fork of the [Retrieval-based Voice Conversion project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion).

> [!NOTE]
> I am NOT the original creator of RVC. The original project can be found [here](https://github.com/RVC-Project).

I created this fork because the original code for RVC is pretty awkward to integrate into projects. This library should hopefully make that process simpler for other developers by providing:
1. A dirt simple API for converting voices.
2. Automatic downloads of the required models (thanks to the [Applio Hugginface Repository](https://huggingface.co/IAHispano/Applio)).
3. An easy installation process with all dependencies accounted for.

## Installation

Install the package:
```
pip install git+https://github.com/Nikerino/LibRVC.git
```

Install PyTorch with CUDA support:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```python
from librvc import RVC
import soundfile as sf

rvc = RVC()

data, fs = rvc.convert('speaker.pth', 'speaker.index', 'source.wav', pitch_shift=6)
sf.write('out.wav', data, fs)
```