from importlib import resources

from routeforge_train.download import RemoteFile

LRB_ARCHIVE = RemoteFile(
    "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/"
    "0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz",
    1_283_503_080,
    "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1",
)

SOURCE = {
    "dataset": "NPULH/LLMRouterBench",
    "revision": "0e5af1b84bf73437a01a1849c0f1d2468baa93fc",
}

# Pool order is also the policy model order and the tie-break order.
POOL = (
    "gpt-5",
    "claude-sonnet-4",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "deepseek-v3.1-terminus",
    "qwen3-235b-a22b-2507",
)

DATASET_FAMILY = {
    "aime": "reasoning",
    "livemathbench": "reasoning",
    "gpqa": "reasoning",
    "mmlupro": "reasoning",
    "livecodebench": "code",
    "simpleqa": "qa",
    "hle": "qa",
    "arenahard": "instruction",
}

_MEMBERS_FILE = "llmrouterbench_members.txt"


def load_members() -> list[str]:
    """Exact archive member names to extract, one per line in the package data file."""
    text = resources.files("routeforge_train").joinpath(_MEMBERS_FILE).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
