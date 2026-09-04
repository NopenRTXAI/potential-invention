"""Low-disk Hugging Face streaming adapter.

This module never calls download() and never materializes a whole dataset.
It yields cleaned dialogues in small batches. Dataset schemas vary, so the
adapter accepts common text, dialogue, messages, and conversations fields.
"""

import re

_TEXT_KEYS = ("text", "content", "utterance", "response")
_DIALOGUE_KEYS = ("dialogue", "conversation", "conversations", "messages")


def clean_text(value):
    """Normalize one text value and discard empty/control-heavy content."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) < 2 or len(text) > 12000:
        return ""
    if sum(ch.isalnum() for ch in text) < 2:
        return ""
    return text


def _message_text(item):
    if isinstance(item, str):
        return clean_text(item)
    if isinstance(item, dict):
        for key in ("content", "text", "value", "utterance"):
            text = clean_text(item.get(key))
            if text:
                return text
    return ""


def record_to_dialogue(record):
    """Convert a HF record to a list of alternating utterances."""
    for key in _DIALOGUE_KEYS:
        value = record.get(key)
        if isinstance(value, (list, tuple)):
            dialogue = [_message_text(item) for item in value]
            dialogue = [item for item in dialogue if item]
            if len(dialogue) >= 2:
                return dialogue
    values = [clean_text(record.get(key)) for key in _TEXT_KEYS]
    values = [value for value in values if value]
    if len(values) >= 2:
        return values
    if values:
        return values
    return []


def stream_dialogues(dataset_id, split="train", config=None, token=None,
                     max_dialogues=None, batch_size=128):
    """Yield cleaned dialogue batches from Hugging Face Hub.

    Requires the `datasets` package. `streaming=True` keeps examples remote;
    `max_dialogues` is a guardrail for experiments and disk-free training.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install streaming support with: pip install datasets") from exc
    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    stream = load_dataset(dataset_id, **kwargs)
    batch = []
    count = 0
    for record in stream:
        dialogue = record_to_dialogue(record)
        if len(dialogue) < 2:
            continue
        batch.append(dialogue)
        count += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if max_dialogues is not None and count >= max_dialogues:
            break
    if batch:
        yield batch


def stream_stats(dataset_id, split="train", config=None, token=None,
                 max_dialogues=1000):
    """Inspect a stream without downloading it or consuming all of it."""
    dialogues = 0
    utterances = 0
    for batch in stream_dialogues(dataset_id, split, config, token,
                                  max_dialogues=max_dialogues):
        dialogues += len(batch)
        utterances += sum(len(dialogue) for dialogue in batch)
    return {"dataset": dataset_id, "split": split,
            "dialogues_sampled": dialogues,
            "utterances_sampled": utterances,
            "streaming": True}
