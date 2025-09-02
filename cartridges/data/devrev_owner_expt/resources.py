import random
from typing import List
import json

from cartridges.data.resources import Resource
import pandas as pd


class RoutingResource(Resource):

    class Config(Resource.Config):
        path: str
        seed: int = 42
        prompts: List[str] = []

    def __init__(self, config: Config):
        self.config = config

        random.seed(self.config.seed)
        self.entries: List[dict] = self._parse_csv(self.config.path)
        # Store all prompt types for selection based on file type
        self.prompts = PROMPTS
        self.default_prompts = PROMPTS if not self.config.prompts else self.config.prompts


    def _parse_csv(self, path: str) -> List[dict]:
        """Parse the Client Discovery Enriched CSV file and group by repository.

        Returns a list of dictionaries where each dictionary represents a
        repository-level chunk with all rows for that repository under 'rows'.
        """
        df = pd.read_csv(path)
        # Drop unneeded columns if present
        columns_to_drop = [
            'Author ID',
            'Reviewer ID',
            'Last Updated Date',
            'Commits (Last 20 days)'
        ]
        drop_existing = [col for col in columns_to_drop if col in df.columns]
        if drop_existing:
            df = df.drop(columns=drop_existing)
        # Group by Parent Repository so each entry is a repository chunk
        grouped_entries: List[dict] = []
        for repo, group_df in df.groupby('Parent Repository', dropna=False):
            grouped_entries.append({
                'Parent Repository': repo,
                'rows': group_df.to_dict('records')
            })
        return grouped_entries

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        # Calculate weights based on number of rows per repository
        weights = [len(entry['rows']) for entry in self.entries]
        entry = random.choices(self.entries, weights=weights, k=1)[0]
        # Join all fields in entry to get the chunk text
        chunk_text = json.dumps(entry)

        # If custom prompts are provided in config, use those
        if self.config.prompts:
            selected_prompts = self.default_prompts
        else:
            # Otherwise, select prompts based on file type
            selected_prompts = self.prompts

        prompts = [prompt for prompt in random.choices(selected_prompts, k=batch_size)]
        return chunk_text, prompts


PROMPTS = [
    "Ask: What all RPCs are there in the repository? Require listing every RPC name. Ensure the question is answerable from the provided context.",
    "Ask: For one specific RPC in the repository, who is its author and who is its reviewer? Require naming both and the RPC. Ensure the question is answerable from the provided context.",
    "Ask: For one specific RPC in the repository, what does it do based on its description? Require naming the RPC and summarizing the description. Ensure the question is answerable from the provided context.",
    "Ask: How many RPCs are in the repository, and what are their names? Ensure the question is answerable from the provided context.",
    "Ask: List all unique authors in the repository along with the count of RPCs attributed to each. Ensure the question is answerable from the provided context.",
    "Ask: List all unique reviewers in the repository along with the count of RPCs they reviewed. Ensure the question is answerable from the provided context."
]

if __name__ == "__main__":
    import asyncio

    async def _run() -> None:
        config = RoutingResource.Config(path="examples/devrev_owner_expt/client_discovery_enriched.csv")
        resource = RoutingResource(config)
        chunk_text, prompts = await resource.sample_prompt(1)
        print(chunk_text)
        print(prompts)

    asyncio.run(_run())