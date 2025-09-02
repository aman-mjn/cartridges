import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient

# PARALLELIZATION SCALING OPTIONS:
# - Conservative: max_num_batches_in_parallel=8  → 256 conversations parallel
# - Maximum:      max_num_batches_in_parallel=256 → 8,192 conversations parallel (current setting)
# - Modal config: 32 pods × 8 concurrent inputs × 32 batch_size = 8,192 conversations
# - With 65,536 samples: 8 iterations at max scale vs 256 iterations at conservative scale
# from cartridges.clients.openai import OpenAIClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
from cartridges.data.devrev_owner_expt.resources import RoutingResource


# config = OpenAIClient.Config(
#     model_name="Qwen/Qwen3-4B-Instruct-2507",  # HF model
#     base_url="http://127.0.0.1:8085/v1",
# )
# client = OpenAIClient(config)

client = TokasaurusClient.Config(
    url="https://pratham1002--toka-qwen3-4b-2507-1xh100-serve.modal.run",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
)

enriched_data_path = os.path.join(os.environ["CARTRIDGES_DIR"], "examples/devrev_owner_expt/client_discovery_enriched.csv")


config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        resources=[
            RoutingResource.Config(
                path=enriched_data_path,
            )
        ],
        max_completion_tokens_a=2048,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=16384, # total number of question-answer pairs to be generated
    batch_size=1, # number of question-answer pairs to be generated in parallel
    max_num_batches_in_parallel=32*8*2,  # MAX SCALE: 8 pods x 8 concurrent x 32 batch_size = 2048 conversations parallel (8 iterations total)
    name=FormatStringVariable(f"{Path(__file__).stem}_n{ '{num_samples}' }"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        tags=["code_synthesis"],
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__":
    pydrantic.main([config])