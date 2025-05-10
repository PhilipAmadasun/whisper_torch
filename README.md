# whisper_testing

### Library Versions
The following library versions were used for this work:
-    torch                             2.6.0
-    torchaudio                        2.6.0
-    transformers                      4.47.1
To test and compare inference with original whisper implementation:
```
python3 test.py <audio.wav>
```
Torch model currently produces wildy inaccurate inference on shorter audio clips as beam search and other optimizations are not yet implemented.
