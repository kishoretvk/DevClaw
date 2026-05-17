"""
MTP (Multi-Token Prediction) module.

Implements Medusa-style auxiliary prediction heads that predict K tokens
simultaneously from the same hidden state, replacing sequential drafting
with parallel prediction for faster speculative decoding.

Reference: "Medusa: Simple LLM Inference Acceleration Framework with
Multiple Decoding Heads" (Cai et al., 2024)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class MedusaHead(nn.Module):
    """Single auxiliary prediction head for one future position."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict logits for a future position from hidden states."""
        return self.head(hidden_states)


class MedusaMTP(nn.Module):
    """
    Medusa-style Multi-Token Prediction.

    Adds K auxiliary heads that each predict a different future token
    from the SAME hidden state. This replaces sequential draft decoding
    with parallel prediction.

    For K=5: predicts tokens at positions [t+1, t+2, t+3, t+4, t+5]
    all from the hidden state at position t.
    """

    def __init__(self, model, num_heads: int = 5):
        """
        Args:
            model: The target model to add heads to
            num_heads: Number of auxiliary heads (K tokens to predict)
        """
        super().__init__()
        self.model = model
        self.num_heads = num_heads

        # Get model dimensions
        config = model.config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Create auxiliary heads (head 0 is the original lm_head)
        self.heads = nn.ModuleList([
            MedusaHead(self.hidden_size, self.vocab_size)
            for _ in range(num_heads)
        ])

        # Move heads to same device as model
        device = next(model.parameters()).device
        self.heads = self.heads.to(device)

    def predict_parallel(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Predict K future tokens in parallel from the same hidden state.

        Args:
            hidden_states: Model hidden states (batch, seq_len, hidden_size)

        Returns:
            List of K logit tensors, each (batch, seq_len, vocab_size)
        """
        # Use last token's hidden state for prediction
        last_hidden = hidden_states[:, -1:, :]  # (batch, 1, hidden_size)

        # All heads predict in parallel from the same hidden state
        all_logits = []
        for head in self.heads:
            logits = head(last_hidden)  # (batch, 1, vocab_size)
            all_logits.append(logits)

        return all_logits

    def draft_parallel(self, input_ids: torch.Tensor,
                       temperature: float = 0.0) -> List[int]:
        """
        Draft K tokens in parallel using auxiliary heads.

        Args:
            input_ids: Current token sequence (batch, seq_len)
            temperature: Sampling temperature (0 = greedy)

        Returns:
            List of K drafted token IDs
        """
        with torch.no_grad():
            # Get hidden states from the model
            outputs = self.model(input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

            # Predict K tokens in parallel
            all_logits = self.predict_parallel(hidden_states)

            # Sample from each head
            draft_tokens = []
            for logits in all_logits:
                logits = logits[:, -1, :]  # (batch, vocab_size)
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                else:
                    token = torch.argmax(logits, dim=-1, keepdim=True)
                draft_tokens.append(token.item())

        return draft_tokens

    def verify(self, input_ids: torch.Tensor, draft_tokens: List[int],
               past_kv=None, threshold: float = 0.1) -> Tuple[List[int], int, tuple]:
        """
        Verify drafted tokens against the target model.

        Args:
            input_ids: Current token sequence (batch, seq_len)
            draft_tokens: List of drafted token IDs
            past_kv: Past key values (optional)
            threshold: Acceptance threshold

        Returns:
            accepted_tokens, num_accepted, new_past_kv
        """
        if not draft_tokens:
            return [], 0, past_kv

        with torch.no_grad():
            # Concatenate input with draft tokens
            draft_tensor = torch.tensor(draft_tokens, device=input_ids.device).unsqueeze(0)
            verify_input = torch.cat([input_ids, draft_tensor], dim=1)

            # Single forward pass through target model
            outputs = self.model(verify_input, past_key_values=past_kv, use_cache=True)
            target_logits = outputs.logits
            new_past_kv = outputs.past_key_values

        # Accept contiguous prefix
        accepted = []
        offset = input_ids.shape[1]
        for i, draft_token in enumerate(draft_tokens):
            pos = offset + i - 1
            if pos >= target_logits.shape[1]:
                break

            target_probs = torch.softmax(target_logits[:, pos, :], dim=-1)
            draft_prob = target_probs[0, draft_token].item()

            if draft_prob > threshold:
                accepted.append(draft_token)
            else:
                break

        return accepted, len(accepted), new_past_kv
