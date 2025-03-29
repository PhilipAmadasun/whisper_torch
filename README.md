# whisper_torch
An implementation of whisper `turbo` model in torch. Uses the pretrained weights for the `turbo` model.

### Library Versions
The following library versions were used for this work:
-    torch                             2.6.0
-    torchaudio                        2.6.0

To test and compare inference with original whisper implementation:
```
python3 test.py <audio.wav>
```
Torch model currently produces wildy inaccurate inference on shorter audio clips.
