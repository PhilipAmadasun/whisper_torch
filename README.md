# whisper_torch
An implementation of whisper `turbo` model in torch (class and feed forward method). Uses the pretrained weights for the `turbo` model.

### Library Versions
The following library versions were used for this work:
-    torch                             2.6.0
-    torchaudio                        2.6.0
-    transformers                      4.47.1

### Transfer weights and create .pt file 
```python
from whisper_turbo_implementation import WhisperTurbo
WhisperTurbo(save_after_init=True, weights_path='whisper_turbo_128mel.pt')
```
### use .pt weights
```python
from whisper_turbo_implementation import WhisperTurbo
turbo = WhisperTurbo(weights_path="whisper_turbo_128mel.pt")
print(turbo.transcribe("../test_wav_files/test_audio1.wav", beam_size=1))
```

