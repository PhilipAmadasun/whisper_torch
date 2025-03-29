import os
import torch
import whisper
from tokenizer import get_tokenizer
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper_model import Whisper, ModelDimensions
import numpy as np

class WhisperTurbo:
    """
    A wrapper class for the Whisper "turbo" model implementation
    that handles all preprocessing and generation
    """
    
    def __init__(self, device=None):
        """
        Initialize the model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load the official model to get dimensions and weights
        print("Loading official Whisper 'turbo' model...")
        self.official_model = whisper.load_model("turbo")  # Use turbo directly if available
        
        # Get model dimensions from the official model
        dims = ModelDimensions(
            n_mels=80,  # Whisper's log_mel_spectrogram produces 80 mel bands
            n_audio_ctx=self.official_model.dims.n_audio_ctx,
            n_audio_state=self.official_model.dims.n_audio_state,
            n_audio_head=self.official_model.dims.n_audio_head,  # Use exact value from official model
            n_audio_layer=self.official_model.dims.n_audio_layer,
            n_text_ctx=self.official_model.dims.n_text_ctx,
            n_text_state=self.official_model.dims.n_text_state,
            n_text_head=self.official_model.dims.n_text_head,  # Use exact value from official model
            n_text_layer=self.official_model.dims.n_text_layer,
            n_vocab=self.official_model.dims.n_vocab
        )
        
        # Debugging: Print model dimensions
        print(f"Model dimensions: {dims}")
        
        # Create and load the custom model
        print("Creating custom Whisper model...")
        self.custom_model = Whisper(dims)
        
        # Transfer weights from official to custom model
        print("Transferring weights...")
        self._transfer_weights()
        
        # Move model to device
        self.custom_model = self.custom_model.to(device)
        self.tokenizer = get_tokenizer(multilingual=True)
        
        # Set model to evaluation mode
        self.custom_model.eval()
    
    def _print_token_info(self):
        """Print information about special tokens"""
        print("\nTokenizer Information:")
        # The Whisper tokenizer doesn't have a 'tokenizer' attribute directly
        # It's either accessing the internal model or using other properties
        
        # Print some special tokens
        special_tokens = {
            "SOT": 50258,  # Start of transcript
            "EOT": 50257,  # End of text
            "EN": 50259,   # English token
            "Transcribe": 50358,
            "Translate": 50359,
            "SOT+EN+Transcribe": [50258, 50259, 50358]  # Common pattern
        }
        
        for name, token_id in special_tokens.items():
            if isinstance(token_id, list):
                decoded = self.tokenizer.decode(token_id)
                print(f"  {name}: {token_id} -> {decoded}")
            else:
                decoded = self.tokenizer.decode([token_id])
                print(f"  {name}: {token_id} -> {decoded}")
        
        # Print some language tokens
        print("\nLanguage Tokens:")
        # This is safer than using LANGUAGES directly with the tokenizer
        language_samples = ["en", "fr", "de", "es", "zh"]
        
        for lang in language_samples:
            try:
                # Use the tokenizer's language token encoding if available
                if hasattr(self.tokenizer, "language_token"):
                    token = self.tokenizer.language_token(lang)
                    token_id = self.tokenizer.encode(token)[-1]
                else:
                    # Fallback to manual encoding
                    token_id = self.tokenizer.encode(f" <|{lang}|>")[-1]
                    
                decoded = self.tokenizer.decode([token_id])
                print(f"  {lang}: {token_id} -> {decoded}")
            except Exception as e:
                print(f"  Error getting token for {lang}: {e}")
    
    def _verify_weight_transfer(self):
        """Verify that weights were transferred correctly by comparing a few tensors"""
        print("\nVerifying weight transfer...")
        
        # Check a few key layers
        layers_to_check = [
            ("encoder.conv2.weight", "encoder.conv2.weight"),
            ("encoder.blocks.0.attn.query.weight", "encoder.blocks.0.attn.query.weight"),
            ("decoder.token_embedding.weight", "decoder.token_embedding.weight"),
            ("decoder.blocks.0.attn.query.weight", "decoder.blocks.0.attn.query.weight")
        ]
        
        official_state = self.official_model.state_dict()
        custom_state = self.custom_model.state_dict()
        
        for official_key, custom_key in layers_to_check:
            if official_key in official_state and custom_key in custom_state:
                # Make sure we're comparing on the same device - convert both to CPU
                official_param = official_state[official_key].cpu()
                custom_param = custom_state[custom_key].cpu()
                
                # Check if shapes match
                shape_match = official_param.shape == custom_param.shape
                
                # Check if values are close
                if shape_match:
                    # Calculate maximum absolute difference
                    max_diff = (official_param - custom_param).abs().max().item()
                    mean_diff = (official_param - custom_param).abs().mean().item()
                    
                    print(f"  {custom_key}: Shape match: {shape_match}, Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                else:
                    print(f"  {custom_key}: Shape mismatch - Official: {official_param.shape}, Custom: {custom_param.shape}")
    
    def _transfer_weights(self):
        """
        Transfer weights from the official model to the custom model
        """
        # Create mapping dictionary
        weight_mapping = {}
        
        # Map encoder convolutional layers (skip first conv layer)
        # weight_mapping["encoder.conv1.weight"] = "encoder.conv1.weight"
        # weight_mapping["encoder.conv1.bias"] = "encoder.conv1.bias"
        weight_mapping["encoder.conv2.weight"] = "encoder.conv2.weight"
        weight_mapping["encoder.conv2.bias"] = "encoder.conv2.bias"
        
        # Map encoder positional embeddings
        weight_mapping["encoder.positional_embedding"] = "encoder.positional_embedding"
        
        # Map encoder transformer blocks
        for i in range(self.custom_model.dims.n_audio_layer):
            # Self-attention
            weight_mapping[f"encoder.blocks.{i}.attn.query.weight"] = f"encoder.blocks.{i}.attn.query.weight"
            weight_mapping[f"encoder.blocks.{i}.attn.query.bias"] = f"encoder.blocks.{i}.attn.query.bias"
            weight_mapping[f"encoder.blocks.{i}.attn.key.weight"] = f"encoder.blocks.{i}.attn.key.weight"
            weight_mapping[f"encoder.blocks.{i}.attn.value.weight"] = f"encoder.blocks.{i}.attn.value.weight"
            weight_mapping[f"encoder.blocks.{i}.attn.value.bias"] = f"encoder.blocks.{i}.attn.value.bias"
            weight_mapping[f"encoder.blocks.{i}.attn.out.weight"] = f"encoder.blocks.{i}.attn.out.weight"
            weight_mapping[f"encoder.blocks.{i}.attn.out.bias"] = f"encoder.blocks.{i}.attn.out.bias"
            
            # Layer norms
            weight_mapping[f"encoder.blocks.{i}.attn_ln.weight"] = f"encoder.blocks.{i}.attn_ln.weight"
            weight_mapping[f"encoder.blocks.{i}.attn_ln.bias"] = f"encoder.blocks.{i}.attn_ln.bias"
            weight_mapping[f"encoder.blocks.{i}.mlp_ln.weight"] = f"encoder.blocks.{i}.mlp_ln.weight"
            weight_mapping[f"encoder.blocks.{i}.mlp_ln.bias"] = f"encoder.blocks.{i}.mlp_ln.bias"
            
            # MLP layers
            weight_mapping[f"encoder.blocks.{i}.mlp.0.weight"] = f"encoder.blocks.{i}.mlp.0.weight"
            weight_mapping[f"encoder.blocks.{i}.mlp.0.bias"] = f"encoder.blocks.{i}.mlp.0.bias"
            weight_mapping[f"encoder.blocks.{i}.mlp.2.weight"] = f"encoder.blocks.{i}.mlp.2.weight"
            weight_mapping[f"encoder.blocks.{i}.mlp.2.bias"] = f"encoder.blocks.{i}.mlp.2.bias"
        
        # Map encoder layer norm
        weight_mapping["encoder.ln_post.weight"] = "encoder.ln_post.weight"
        weight_mapping["encoder.ln_post.bias"] = "encoder.ln_post.bias"
        
        # Map decoder embeddings
        weight_mapping["decoder.token_embedding.weight"] = "decoder.token_embedding.weight"
        weight_mapping["decoder.positional_embedding"] = "decoder.positional_embedding"
        
        # Map decoder transformer blocks
        for i in range(self.custom_model.dims.n_text_layer):
            # Self-attention
            weight_mapping[f"decoder.blocks.{i}.attn.query.weight"] = f"decoder.blocks.{i}.attn.query.weight"
            weight_mapping[f"decoder.blocks.{i}.attn.query.bias"] = f"decoder.blocks.{i}.attn.query.bias"
            weight_mapping[f"decoder.blocks.{i}.attn.key.weight"] = f"decoder.blocks.{i}.attn.key.weight"
            weight_mapping[f"decoder.blocks.{i}.attn.value.weight"] = f"decoder.blocks.{i}.attn.value.weight"
            weight_mapping[f"decoder.blocks.{i}.attn.value.bias"] = f"decoder.blocks.{i}.attn.value.bias"
            weight_mapping[f"decoder.blocks.{i}.attn.out.weight"] = f"decoder.blocks.{i}.attn.out.weight"
            weight_mapping[f"decoder.blocks.{i}.attn.out.bias"] = f"decoder.blocks.{i}.attn.out.bias"
            
            # Cross-attention
            weight_mapping[f"decoder.blocks.{i}.cross_attn.query.weight"] = f"decoder.blocks.{i}.cross_attn.query.weight"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.query.bias"] = f"decoder.blocks.{i}.cross_attn.query.bias"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.key.weight"] = f"decoder.blocks.{i}.cross_attn.key.weight"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.value.weight"] = f"decoder.blocks.{i}.cross_attn.value.weight"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.value.bias"] = f"decoder.blocks.{i}.cross_attn.value.bias"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.out.weight"] = f"decoder.blocks.{i}.cross_attn.out.weight"
            weight_mapping[f"decoder.blocks.{i}.cross_attn.out.bias"] = f"decoder.blocks.{i}.cross_attn.out.bias"
            
            # Layer norms
            weight_mapping[f"decoder.blocks.{i}.attn_ln.weight"] = f"decoder.blocks.{i}.attn_ln.weight"
            weight_mapping[f"decoder.blocks.{i}.attn_ln.bias"] = f"decoder.blocks.{i}.attn_ln.bias"
            weight_mapping[f"decoder.blocks.{i}.cross_attn_ln.weight"] = f"decoder.blocks.{i}.cross_attn_ln.weight"
            weight_mapping[f"decoder.blocks.{i}.cross_attn_ln.bias"] = f"decoder.blocks.{i}.cross_attn_ln.bias"
            weight_mapping[f"decoder.blocks.{i}.mlp_ln.weight"] = f"decoder.blocks.{i}.mlp_ln.weight"
            weight_mapping[f"decoder.blocks.{i}.mlp_ln.bias"] = f"decoder.blocks.{i}.mlp_ln.bias"
            
            # MLP layers
            weight_mapping[f"decoder.blocks.{i}.mlp.0.weight"] = f"decoder.blocks.{i}.mlp.0.weight"
            weight_mapping[f"decoder.blocks.{i}.mlp.0.bias"] = f"decoder.blocks.{i}.mlp.0.bias"
            weight_mapping[f"decoder.blocks.{i}.mlp.2.weight"] = f"decoder.blocks.{i}.mlp.2.weight"
            weight_mapping[f"decoder.blocks.{i}.mlp.2.bias"] = f"decoder.blocks.{i}.mlp.2.bias"
        
        # Map decoder layer norm
        weight_mapping["decoder.ln.weight"] = "decoder.ln.weight"
        weight_mapping["decoder.ln.bias"] = "decoder.ln.bias"
        
        # Create new state dict with transferred weights
        new_state_dict = {}
        missing_keys = []
        
        # Print some debug info
        print("\nOfficial model state keys (first 5):")
        for i, key in enumerate(list(self.official_model.state_dict().keys())[:5]):
            print(f"  {key}")
        
        print("\nCustom model state keys (first 5):")
        for i, key in enumerate(list(self.custom_model.state_dict().keys())[:5]):
            print(f"  {key}")
        
        # Perform the transfer
        for custom_key, official_key in weight_mapping.items():
            if official_key in self.official_model.state_dict():
                new_state_dict[custom_key] = self.official_model.state_dict()[official_key]
            else:
                missing_keys.append(official_key)
        
        # Special handling for the first convolutional layer
        try:
            official_conv1_weight = self.official_model.state_dict()["encoder.conv1.weight"]
            official_conv1_bias = self.official_model.state_dict()["encoder.conv1.bias"]
            
            # If original model used 128 channels and we need 80, we can take the first 80 channels
            if official_conv1_weight.shape[1] > 80:
                print("Adapting encoder.conv1 weights to new input dimensions...")
                new_state_dict["encoder.conv1.weight"] = official_conv1_weight[:, :80, :]
                new_state_dict["encoder.conv1.bias"] = official_conv1_bias
            else:
                print("Cannot adapt conv1 weights, dimensions are incompatible")
        except (KeyError, IndexError) as e:
            print(f"Error adapting conv1 weights: {e}")
        
        # Load the new state dict
        missing_custom, unexpected = self.custom_model.load_state_dict(new_state_dict, strict=False)
        
        print(f"\nMissing keys in transfer: {len(missing_keys)}")
        print(f"Missing keys in custom model: {len(missing_custom)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        if len(missing_custom) > 0:
            print(f"First few missing custom keys: {missing_custom[:3]}")
    
    def transcribe(self, audio_path, language="en", task="transcribe", verbose=False):
        """
        Transcribe audio using the custom model
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use whisper's preprocessing to ensure proper input format
        audio = load_audio(audio_path)
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio)
        mel = torch.unsqueeze(mel, 0).to(self.device)
        
        if verbose:
            print(f"Audio shape: {audio.shape}")
            print(f"Mel shape: {mel.shape}")
        
        # For more accurate comparison, try to mimic how the official model works
        try:
            # First attempt: Use the official model's approach but with our model
            options = {
                "language": language,
                "task": task,
                # Add other options that might affect the generation
            }
            
            # Generate tokens
            with torch.no_grad():
                tokens = self.custom_model.generate(
                    mel, 
                    language=language,
                    task=task,
                    temperature=0.0  # Use greedy decoding
                )
                
                if verbose:
                    print(f"Generated tokens: {tokens}")
                
                # Decode the tokens
                transcription = self._decode_tokens(tokens)
                
            return transcription
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            # Fallback to a simpler approach if there's an error
            return "Error in transcription"
    
    def _decode_tokens(self, tokens):
        """
        Use the official logic from tokenizer.py:
          - All tokens >= self.tokenizer.timestamp_begin are removed.
          - The remaining tokens are merged with the official merges.
        """
        # The official tokenizer's decode() already does the filtering for timestamps.
        # So we just call it:
        return self.tokenizer.decode(tokens).strip()

    
    def compare_with_official(self, audio_path):
        """
        Compare transcription with the official Whisper model
        """
        # Transcribe with custom model
        custom_transcription = self.transcribe(audio_path)
        
        # Use official model's own transcribe method
        print("\nRunning official model...")
        result = self.official_model.transcribe(audio_path)
        official_transcription = result["text"]
        
        print("\nComparison:")
        print(f"Custom:   \"{custom_transcription}\"")
        print(f"Official: \"{official_transcription}\"")

        print(f"Official tokens: {result["tokens"]}")
        # Also show the token sequences
        audio = load_audio(audio_path)
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio)
        mel = torch.unsqueeze(mel, 0).to(self.device)
        
        # Get sequences from both models for direct comparison
        with torch.no_grad():
            # Get custom tokens
            custom_tokens = self.custom_model.generate(mel, language="en", task="transcribe")
            print(f"\nCustom tokens: {custom_tokens}")
            
            # Get official tokens (by using the same generate function if possible)
            try:
                # Try to access the internal generate function of the official model
                if hasattr(self.official_model, 'generate'):
                    # If the official model has a generate method, use it
                    official_tokens = self.official_model.generate(mel, language="en", task="transcribe")
                    print(f"Official tokens: {official_tokens}")
                    
                    # Compare token sequences
                    print("\nToken Comparison:")
                    if len(custom_tokens) > 0 and len(official_tokens) > 0:
                        for i in range(min(len(custom_tokens), len(official_tokens), 10)):  # Compare first 10 tokens
                            custom_token = custom_tokens[i] if i < len(custom_tokens) else None
                            official_token = official_tokens[i] if i < len(official_tokens) else None
                            print(f"Position {i}: Custom={custom_token}, Official={official_token}")
            except Exception as e:
                print(f"Could not get official tokens for comparison: {e}")
        
        return custom_transcription, official_transcription


def main():
    # Example usage
    audio_path = "test_audio.wav"  # Replace with your test audio file
    
    # Create model
    model = WhisperTurbo()
    
    # Transcribe audio
    if os.path.exists(audio_path):
        print(f"\nTranscribing {audio_path}...")
        transcription = model.transcribe(audio_path, verbose=True)
        print(f"Transcription: {transcription}")
        
        # Compare with official model
        model.compare_with_official(audio_path)
    else:
        print(f"Test audio file {audio_path} not found. Please specify a valid audio file.")

if __name__ == "__main__":
    main()
