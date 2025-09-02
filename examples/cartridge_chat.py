#!/usr/bin/env python3
"""
Interactive CLI for chatting with a trained cache model.
Usage: python examples/cartridge_chat.py <config_path> <cache_path>
Example: python examples/cartridge_chat.py outputs/2025-08-27-05-51-52-code_train/6121e226-55b6-47ae-aca9-cd95613089a1/config.yaml outputs/2025-08-27-05-51-52-code_train/6121e226-55b6-47ae-aca9-cd95613089a1/cache-step191.pt
"""

import argparse
import sys
import os
import torch
import readline
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer

from cartridges.train import TrainConfig, CacheAndModel
from cartridges.cache import TrainableCache, AttnConfig
from cartridges.generation import flex_generate


def compose_caches(cache_paths: List[str], device: str = "cuda", dtype=torch.bfloat16) -> TrainableCache:
    """Load and concatenate multiple cache checkpoints into a single, frozen cache.

    Concatenation is along the sequence dimension for each layer's keys/values.
    All tokens are marked as cartridge tokens (frozen) for inference-only chat.
    """
    if len(cache_paths) == 0:
        raise ValueError("compose_caches requires at least one cache path")

    caches = [
        TrainableCache.from_pretrained(p, device=device).to(device).to(dtype)
        for p in cache_paths
    ]

    cfg = caches[0].config
    for c in caches[1:]:
        assert (
            c.config.n_layers == cfg.n_layers
            and c.config.n_heads == cfg.n_heads
            and c.config.head_dim == cfg.head_dim
        ), "All caches must have matching attention configuration"

    init_keys: List[torch.Tensor] = []
    init_values: List[torch.Tensor] = []
    for layer in range(cfg.n_layers):
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        for c in caches:
            if getattr(c, "_num_frozen_tokens", 0) > 0:
                parts_k.append(c.frozen_keys[layer])
                parts_v.append(c.frozen_values[layer])
            if getattr(c, "_num_trainable_tokens", 0) > 0:
                parts_k.append(c.trainable_keys[layer])
                parts_v.append(c.trainable_values[layer])
        init_keys.append(torch.cat(parts_k, dim=2).contiguous())
        init_values.append(torch.cat(parts_v, dim=2).contiguous())

    total_tokens = init_keys[0].shape[2]
    combined = TrainableCache(
        config=AttnConfig(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
        ),
        init_keys=init_keys,
        init_values=init_values,
        num_frozen_tokens=total_tokens,
    ).to(device).to(dtype)
    return combined


class ChatSession:
    def __init__(self, model, tokenizer, cache):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.cache_paths: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.input_history: List[str] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def undo_last_message(self):
        """Remove the last two messages (user and assistant)."""
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]
            return True
        elif len(self.conversation_history) == 1:
            self.conversation_history = []
            return True
        return False
    
    def clear_conversation(self):
        """Clear the entire conversation history."""
        self.conversation_history = []
    
    def _normalize_path(self, path: str) -> str:
        path = path.strip().strip("'\"")
        return os.path.abspath(path)
    
    def _rebuild_cache(self):
        if len(self.cache_paths) == 0:
            # No cartridges → talk to base model (flex_generate will allocate ephemeral cache)
            self.cache = None
        else:
            self.cache = compose_caches(self.cache_paths, device="cuda", dtype=torch.bfloat16)
    
    def add_cache_path(self, path: str) -> str:
        full = self._normalize_path(path)
        if full in self.cache_paths:
            return f"Cache already added: {full}"
        if not Path(full).exists():
            return f"Cache path does not exist: {full}"
        try:
            self.cache_paths.append(full)
            self._rebuild_cache()
            return f"Added cache: {full}. Total caches: {len(self.cache_paths)}"
        except Exception as e:
            # rollback on failure
            if full in self.cache_paths:
                self.cache_paths.remove(full)
            return f"Failed to add cache: {full}. Error: {e}"
    
    def remove_cache_path(self, path: str) -> str:
        spec = path.strip().lower()
        if spec in {"all", "*"}:
            self.cache_paths = []
            self._rebuild_cache()
            return "Removed all caches. Now chatting with the base model."
        full = self._normalize_path(path)
        if full not in self.cache_paths:
            return f"Cache not found: {full}"
        self.cache_paths.remove(full)
        self._rebuild_cache()
        if len(self.cache_paths) == 0:
            return f"Removed cache: {full}. No caches remain; using base model."
        return f"Removed cache: {full}. Total caches: {len(self.cache_paths)}"
    
    def generate_response(self, user_input: str, enable_thinking: bool = False) -> str:
        """Generate a response to the user input."""
        # Add to input history for readline
        if user_input.strip() and user_input not in self.input_history:
            self.input_history.append(user_input)
            readline.add_history(user_input)
        
        # Add user message to history
        self.add_message("user", user_input)
        
        # Prepare the conversation for the tokenizer
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to("cuda")
        
        # Flatten and create seq_ids and position_ids for single conversation
        flat_input_ids = input_ids.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device="cuda")
        position_ids = torch.arange(flat_input_ids.shape[0], device="cuda")
        
        # Generate response
        output = flex_generate(
            model=self.model,
            input_ids=flat_input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            tokenizer=self.tokenizer,
            cache=self.cache,
            max_new_tokens=1024,
            temperature=0.0,
            show_progress=True,
        )
        
        # Decode the response
        if 0 in output and output[0]:
            response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.add_message("assistant", response_text)
            return response_text
        else:
            return "I'm sorry, I couldn't generate a response."


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained cache model")
    parser.add_argument("config_path", help="Path to config.yaml file")
    parser.add_argument("cache_path", help="Path to cache-*.pt file")
    parser.add_argument(
        "--cache_paths",
        nargs="+",
        help="Additional cache-*.pt files to compose (space- or comma-separated)",
    )
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking")
    args = parser.parse_args()
    
    print(f"Loading model from config: {args.config_path}")
    print(f"Loading cache from: {args.cache_path}")
    
    # Load config and instantiate model
    train_config = TrainConfig.from_yaml(args.config_path, strict=False)
    model = train_config.model.instantiate().to("cuda").to(torch.bfloat16)
    
    # Load cache(s) and ensure it's on cuda with correct dtype
    cache_paths: List[str] = [args.cache_path]
    if args.cache_paths:
        # Support space-separated and comma-separated lists
        extra: List[str] = []
        for item in args.cache_paths:
            extra.extend([p for p in item.split(",") if p])
        cache_paths.extend(extra)
        print(f"Composing {len(cache_paths)} caches:")
        for p in cache_paths:
            print(f" - {p}")
        cache = compose_caches(cache_paths, device="cuda", dtype=torch.bfloat16)
    else:
        cache = TrainableCache.from_pretrained(args.cache_path, device="cuda")
        cache = cache.to("cuda").to(torch.bfloat16)
    
    # Get tokenizer from model config
    tokenizer = AutoTokenizer.from_pretrained(train_config.model.pretrained_model_name_or_path)
    
    print("Model and cache loaded successfully!\n")
    
    # Initialize chat session
    chat = ChatSession(model, tokenizer, cache)
    
    # Configure readline for better input handling
    readline.set_startup_hook(None)
    
    print("=== Chat with Trained Cache ===")
    print("Commands:")
    print("  /undo  - Undo the last message exchange")
    print("  /clear - Clear the entire conversation")
    print("  /add cache <path> - Add a cache at runtime; compose with existing")
    print("  /remove cache <path|all> - Remove a specific cache or all to use base model")
    print("  /quit  - Exit the chat")
    print("  /help  - Show this help message")
    print("Arrow keys: ↑↓ for command history, ←→ for line editing")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input == "/quit":
                print("Goodbye!")
                break
            elif user_input == "/help":
                print("\nCommands:")
                print("  /undo  - Undo the last message exchange")
                print("  /clear - Clear the entire conversation")
                print("  /add cache <path> - Add a cache at runtime; compose with existing")
                print("  /remove cache <path|all> - Remove a specific cache or all to use base model")
                print("  /quit  - Exit the chat")
                print("  /help  - Show this help message")
                print("Arrow keys: ↑↓ for command history, ←→ for line editing")
                continue
            elif user_input == "/undo":
                if chat.undo_last_message():
                    print("Last message exchange undone.")
                else:
                    print("No messages to undo.")
                continue
            elif user_input == "/clear":
                chat.clear_conversation()
                print("Conversation cleared.")
                continue
            elif user_input.startswith("/add "):
                parts = user_input.split(None, 2)
                if len(parts) >= 3 and parts[1] == "cache":
                    msg = chat.add_cache_path(parts[2])
                    print(msg)
                else:
                    print("Usage: /add cache <path>")
                continue
            elif user_input.startswith("/remove "):
                parts = user_input.split(None, 2)
                if len(parts) >= 3 and parts[1] == "cache":
                    msg = chat.remove_cache_path(parts[2])
                    print(msg)
                else:
                    print("Usage: /remove cache <path|all>")
                continue
            
            # Generate and display response
            print("Assistant: ", end="", flush=True)
            response = chat.generate_response(user_input, enable_thinking=args.enable_thinking)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Type /help for available commands.")


if __name__ == "__main__":
    main()
