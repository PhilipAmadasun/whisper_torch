#!/usr/bin/env python3
"""
Simple example showing how to use the WhisperTurbo model for audio transcription
"""

import os
import sys
from whisper_turbo_implementation import WhisperTurbo

def main():
    # Check if an audio file was provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "test_audio.wav"  # Default test file
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        print(f"Usage: {sys.argv[0]} [audio_file_path]")
        return 1
    
    print(f"Transcribing audio file: {audio_path}")
    
    # Create the WhisperTurbo model
    model = WhisperTurbo()  # This will load the "turbo" model
    
    # Transcribe the audio
    transcription = model.transcribe(audio_path, verbose=True)
    print("\nTranscription result:")
    print(f"{transcription}")
    
    # Compare with official model
    print("\nComparing with official Whisper model:")
    model.compare_with_official(audio_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
