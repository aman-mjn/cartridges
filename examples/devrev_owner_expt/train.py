"""
Code training example.

To run:
    cloudexe --gpuspec EUNH200x1 -- /root/aman-cartridges/.venv/bin/python3 examples/devrev_owner_expt/train.py
"""

import os
from pathlib import Path
import pandas as pd

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset


# Qwen-only model support
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM




def _resolve_data_sources() -> list[str]:
    """Resolve training data sources from environment variables (local only).
    """
    local_path = "/root/aman-cartridges/outputs/2025-09-02-09-00-34-synthesize/synthesize_n16384-0/artifact/dataset.parquet"

    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"DATASET_PATH does not exist: {path}")

    # String path is accepted by TrainDataset and routed through read_conversations
    return [str(path)]


def make_config() -> TrainConfig:
    # num_tokens = 4096
    num_tokens = int(os.environ.get("NUM_TOKENS", 4096))

    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        model_cls=FlexQwen3ForCausalLM,
    )

    data_sources = _resolve_data_sources()
    # Create a small holdout split for periodic evaluation
    def _make_holdout(src_path: str, frac: float = 0.05, seed: int = 42) -> tuple[str, str]:
        src = Path(src_path)
        base = src.with_suffix("")
        holdout_path = f"{base}_holdout.parquet"
        train_path = f"{base}_train.parquet"

        if Path(holdout_path).exists() and Path(train_path).exists():
            return holdout_path, train_path

        df = pd.read_parquet(src_path)
        eval_df = df.sample(frac=frac, random_state=seed)
        train_df = df.drop(eval_df.index)

        eval_df.to_parquet(holdout_path, index=False)
        train_df.to_parquet(train_path, index=False)
        return holdout_path, train_path

    holdout_path, train_path = _make_holdout(data_sources[0], frac=0.05, seed=42)

    config = TrainConfig(
        model=model,
        kv_cache_initializer=KVFromText.Config(max_tokens=num_tokens, text_source="/root/aman-cartridges/examples/devrev_owner_expt/client_discovery_enriched.csv"),
        lr=2e-2,
        epochs=1,
        global_batch_size=4,
        dataset=TrainDataset.Config(
            data_sources=[train_path],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),
        # Periodically evaluate perplexity on the holdout set
        loss_eval_every_n_steps=200,
        loss_evals=[
            LossEvalConfig(
                name="holdout",
                dataset=TrainDataset.Config(
                    data_sources=[holdout_path],
                    top_k_logits=20,
                    packed_seq_length=2048,
                    packing_mode="truncate",
                ),
            )
        ],
        generate_eval_every_n_steps=None,
        generate_evals=[],
        distributed_backend="gloo",
        save_every_n_steps=200,

        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        name=FormatStringVariable(
            "devrev_owner_expt_train_qwen_lr{lr}_toks{kv_cache_initializer.max_tokens}_gbs{global_batch_size}"
        ),

    )

    return config


if __name__ == "__main__":
    cfg = make_config()
    pydrantic.main([cfg])


