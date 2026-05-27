import json
from dataclasses import dataclass
from typing import Dict, Iterator, List

from datasets import Dataset


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@dataclass
class ChatSample:
    system: str
    user: str
    assistant: str


DEFAULT_SYSTEM_PROMPT = (
    "You are a professional industry assistant. "
    "You know finance and technology domain concepts and you respond with accurate, clear, and professional language. "
    "Use correct terminology. If asked to summarize, produce a concise professional summary."
)


def to_instruction_format(raw_item: Dict) -> Dict:
    """Convert raw dataset row into a chat-style prompt.

    Expected raw_item keys (dataset.jsonl):
      - category: 'finance' | 'tech' | 'summary'
      - input: question or document text
      - output: ideal answer
    """

    category = raw_item.get("category")
    inp = raw_item["input"].strip()
    out = raw_item["output"].strip()

    if category == "finance":
        user = f"Finance question:\n{inp}"
    elif category == "tech":
        user = f"Technology question:\n{inp}"
    elif category == "summary":
        user = f"Summarize the following document:\n{inp}"
    else:
        user = inp

    return {
        "system": DEFAULT_SYSTEM_PROMPT,
        "user": user,
        "assistant": out,
    }


def build_hf_dataset(raw_path: str) -> Dataset:
    raw = load_jsonl(raw_path)
    formatted = [to_instruction_format(x) for x in raw]
    return Dataset.from_list(formatted)

