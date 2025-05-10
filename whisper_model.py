import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class ModelDimensions:
    n_mels: int   = 128          # we up‑sampled the front‑end
    n_audio_ctx:  int = 1500     # from the checkpoint shape [1500,1280]
    n_audio_state:int = 1280
    n_audio_head: int = 20       # 1280 / 20  = 64‑d heads
    n_audio_layer:int = 32
    n_text_ctx:   int = 448      # checkpoint shape [448,1280]
    n_text_state: int = 1280
    n_text_head:  int = 20
    n_text_layer: int = 32
    n_vocab:      int = 51866 



class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        
        # Match the exact structure from the Whisper model
        self.query = nn.Linear(n_state, n_state, bias=True)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=True)
        self.out = nn.Linear(n_state, n_state, bias=True)

    def forward(self, x, xa=None, mask=None):
        q = self.query(x)
        
        # Cross-attention
        if xa is not None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k = self.key(x)
            v = self.value(x)
            
        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(2, 3)) * scale

        # Apply mask (for causal/future masking in decoder)
        if mask is not None:
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.n_state)
        return self.out(attn_output)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-5)
        
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = nn.LayerNorm(n_state, eps=1e-5) if cross_attention else None
        
        # Match exact MLP structure from Whisper model
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4, bias=True),  # 1280 -> 5120
            nn.GELU(approximate='none'),
            nn.Linear(n_state * 4, n_state, bias=True)   # 5120 -> 1280
        )
        self.mlp_ln = nn.LayerNorm(n_state, eps=1e-5)

    def forward(self, x, xa=None, mask=None):
        # Self-attention with pre-norm
        x = x + self.attn(self.attn_ln(x), mask=mask)
        
        # Cross-attention if applicable
        if self.cross_attn is not None and xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
            
        # Feed-forward network with pre-norm
        x = x + self.mlp(self.mlp_ln(x))
        
        return x


class AudioEncoder(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv1d(dims.n_mels, dims.n_audio_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dims.n_audio_state, dims.n_audio_state, kernel_size=3, stride=2, padding=1)
        
        # Create positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(dims.n_audio_ctx, dims.n_audio_state))
        torch.nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(dims.n_audio_state, dims.n_audio_head)
            for _ in range(dims.n_audio_layer)
        ])
        
        self.ln_post = nn.LayerNorm(dims.n_audio_state, eps=1e-5)
        
    def forward(self, x: torch.Tensor):
        # x has shape [batch_size, n_mels, time]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        # Transpose to [batch_size, time, n_audio_state]
        x = x.permute(0, 2, 1)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        max_pos_len = self.positional_embedding.shape[0]
        
        # Handle case where sequence length exceeds positional embedding size
        if seq_len > max_pos_len:
            # Truncate the sequence to match positional embedding size
            x = x[:, :max_pos_len, :]
            seq_len = max_pos_len
            
        x = x + self.positional_embedding[:seq_len].unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final layer norm
        x = self.ln_post(x)
        
        return x


class TextDecoder(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        
        self.token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        self.positional_embedding = nn.Parameter(torch.empty(dims.n_text_ctx, dims.n_text_state))
        torch.nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(dims.n_text_state, dims.n_text_head, cross_attention=True)
            for _ in range(dims.n_text_layer)
        ])
        
        self.ln = nn.LayerNorm(dims.n_text_state, eps=1e-5)
        
    def forward(self, x: torch.Tensor, xa: torch.Tensor, mask=None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # x is token indices, shape [batch_size, sequence_length]
        x = self.token_embedding(x)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len].unsqueeze(0)
        
        # Apply transformer blocks with cross-attention
        for block in self.blocks:
            x = block(x, xa, mask=mask)
            
        # Apply final layer norm
        x = self.ln(x)
        
        # Return logits by multiplying with the transposed embedding matrix
        logits = torch.matmul(x, self.token_embedding.weight.transpose(0, 1))
        
        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        
        self.dims = dims
        self.encoder = AudioEncoder(dims)
        self.decoder = TextDecoder(dims)
        
    def forward(self, audio, tokens, mask=None):
        # Encode audio features
        audio_features = self.encoder(audio)
        
        # Decode and generate logits
        logits = self.decoder(tokens, audio_features, mask=mask)
        
        return logits
        
    def generate(self, mel, tokenizer=None, language=None, task="transcribe", 
             beam_size: int = 1,           # 1 == greedy; >1 == beam search
             temperature: float = 0.0,
             max_length: int = 448):
        """A more sophisticated generation method that follows Whisper's behavior"""
        device = next(self.parameters()).device
        
        # Encode mel spectrogram
        audio_features = self.encoder(mel.to(device))
        
        # Special tokens and prompting
        sot_token = 50258  # Start of transcript
        
        # For Whisper, English token is actually 50259 (not SOT)
        # And EOT is 50257 (not 50259)
        eot_token = 50257  # End of text token
        
        # Initialize with start of transcript token
        prompt_tokens = [sot_token]
        
        # Add language token if specified
        if language is not None and tokenizer is not None:
            lang_token = tokenizer.to_language_token(language.lower())
            prompt_tokens.append(lang_token)
        
        # Add task token (transcribe or translate)
        if task == "translate":
            task_token = 50359  # translate token
        else:
            task_token = 50358  # transcribe token
        prompt_tokens.append(task_token)

        # NEW ➜ always finish the prompt with <|startoflm|>
        #     This is exactly what the original OpenAI implementation does.
        #     Because we later chop these prompt tokens off, the user will
        #     never see them in the returned transcript.
        sot_lm = 50360                     # <|startoflm|>
        prompt_tokens.append(sot_lm)  
        prompt_tokens.append(tokenizer.no_timestamps)      

        # Print prompt tokens for debugging
        print(f"Prompt tokens: {prompt_tokens}")
            
        # Initialize generation
        tokens = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        
        # Simple greedy search
        with torch.no_grad():
            beams = [(torch.tensor([prompt_tokens], device=device), 0.0)]  # (tokens, logprob)

            for i in range(max_length - len(prompt_tokens)):
                all_candidates = []
                for seq, seq_logprob in beams:
                    seq_len = seq.shape[1]
                    mask = get_causal_mask(seq_len, device)
                    logits = self.decoder(seq, audio_features, mask=mask)[:, -1]

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        logprobs = torch.log(probs + 1e-12)
                    else:
                        logprobs = torch.log_softmax(logits, dim=-1)

                    topk_logprobs, topk_tokens = torch.topk(logprobs, beam_size, dim=-1)

                    for tok, lp in zip(topk_tokens[0], topk_logprobs[0]):
                        new_seq = torch.cat([seq, tok.view(1, 1)], dim=-1)
                        all_candidates.append((new_seq, seq_logprob + lp.item()))

                # keep the best `beam_size` sequences
                ordered = sorted(all_candidates, key=lambda t: t[1], reverse=True)
                beams = ordered[:beam_size]

                # early‑stop if every beam ended with </|endoftext|>
                if all(b[0][0, -1].item() == eot_token for b in beams):
                    break

            # choose best beam
            tokens = beams[0][0]
        
        # Return the generated token sequence (without the prompt)
        return tokens[0].cpu().tolist()[len(prompt_tokens):]


def get_causal_mask(seq_len, device):
    """
    Create a causal mask for self-attention.
    Sets future positions to -inf (effectively zero probability after softmax).
    """
    # Creating mask that prevents attention to future positions
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    # This is a key difference: Whisper expects masks in a specific format
    # Original line: return mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
    # Corrected format:
    return mask  # Shape: [seq_len, seq_len]
