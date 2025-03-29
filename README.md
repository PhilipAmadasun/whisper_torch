# whisper_torch
An implementation of whisper `turbo` model in torch. Uses the pretrained weights for the `turbo` model.

To test and compare inference with original whisper implementation:
```
python3 test.py <audio.wav>
```
Torch model currently produces wildy inaccurate inference on shorter audio clips.
