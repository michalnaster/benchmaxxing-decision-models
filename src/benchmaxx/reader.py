"""Step 1 and 2 of the recipe: read each decision twice with a frozen LLM and keep what it computed.

The LLM is never trained. For every decision the reader builds both readings (:mod:`benchmaxx.prompt`), runs them in
length-sorted batches under a token budget, and keeps, at each requested layer, the answer vector and one option vector
per option, plus the LLM's probability for each option code at the answer position. The two readings are averaged
into one :class:`~benchmaxx.features.FeatureSet` row.
"""

from collections.abc import Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmaxx.device import default_device
from benchmaxx.features import FeatureSet
from benchmaxx.prompt import EncodedPrompt, PromptBuilder
from benchmaxx.schema import Decision

DEFAULT_LLM = "Qwen/Qwen3.5-4B"
DEFAULT_LAYERS = (12, 20, 28, 32)


class LLMReader:
    """A frozen causal LLM that turns decisions into features.

    Args:
        model_name: a Hugging Face causal LM with a chat template.
        layers: indices into the model's hidden states (0 is the embedding output, the last one is the final,
            normalised layer that the output head reads).
        device: where to run the LLM (default: CUDA, then Apple MPS, then CPU).
        dtype: the LLM's weights and activations.
        token_budget: the most padded tokens in one forward pass.
        max_state_tokens: documents are cut after this many tokens.
    """

    def __init__(self, model_name: str = DEFAULT_LLM, layers: Sequence[int] = DEFAULT_LAYERS,
                 device: torch.device | str | None = None, dtype: torch.dtype = torch.bfloat16,
                 token_budget: int = 12_288, max_state_tokens: int = 8192):
        self.model_name = model_name
        self.layers = tuple(layers)
        self.device = torch.device(device) if device is not None else default_device()
        self.token_budget = token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.lm.requires_grad_(False)
        depth = self.lm.config.get_text_config().num_hidden_layers
        if max(self.layers) > depth:
            raise ValueError(f"layers {self.layers} exceed the model's {depth} layers")
        self.builder = PromptBuilder(self.tokenizer, max_state_tokens=max_state_tokens)

    @torch.no_grad()
    def read(self, decisions: Sequence[Decision], progress: bool = True) -> FeatureSet:
        """Features of ``decisions``, in their order."""
        prompts = [p for d in decisions for p in self.builder.readings(d.state, d.question)]
        answer, options, code_logprobs = self._encode(prompts, progress)
        rows_answer, rows_options, rows_probs, gold = [], [], [], []
        for i, decision in enumerate(decisions):
            first, second = 2 * i, 2 * i + 1
            rows_answer.append(_mean_of_readings(answer[first], answer[second]))
            rows_options.append(_mean_of_readings(options[first], options[second]))
            rows_probs.append(torch.stack([code_logprobs[first].softmax(0), code_logprobs[second].softmax(0)], 1))
            k = decision.question.num_options
            gold.append(torch.tensor(decision.gold) if decision.gold is not None else torch.full((k,), float("nan")))
        sizes = torch.tensor([d.question.num_options for d in decisions])
        return FeatureSet(
            answer=torch.stack(rows_answer), options=torch.cat(rows_options), order_probs=torch.cat(rows_probs),
            offsets=torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
            qtype=torch.tensor([d.question.type.code for d in decisions]), gold=torch.cat(gold).float(),
            records=[{"case_id": d.case_id, "dataset": d.dataset, "option_keys": list(d.question.option_keys)}
                     for d in decisions],
            llm={"model": self.model_name, "layers": list(self.layers)})

    def _encode(self, prompts: list[EncodedPrompt], progress: bool
                ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Per prompt: answer vector ``(L, D)`` and option vectors ``(K, L, D)`` in float16, and the log-probability
        of each option's code ``(K,)``."""
        answer: list[torch.Tensor | None] = [None] * len(prompts)
        options: list[torch.Tensor | None] = [None] * len(prompts)
        logprobs: list[torch.Tensor | None] = [None] * len(prompts)
        pad = self.tokenizer.pad_token_id or 0
        with tqdm(total=len(prompts), desc="reading", unit="prompt", disable=not progress) as bar:
            for batch in _length_batches(prompts, self.token_budget):
                width = max(len(prompts[i].ids) for i in batch)
                ids = torch.full((len(batch), width), pad, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):  # right padding leaves every earlier causal position unchanged
                    ids[row, : len(prompts[i].ids)] = torch.tensor(prompts[i].ids)
                    mask[row, : len(prompts[i].ids)] = 1
                out = self.lm.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                    output_hidden_states=True, use_cache=False)
                rows = torch.arange(len(batch), device=self.device)
                positions = torch.tensor([prompts[i].answer_position for i in batch], device=self.device)
                final = out.hidden_states[-1][rows, positions]  # what the output head reads
                token_logprobs = self.lm.get_output_embeddings()(final).float().log_softmax(-1).cpu()
                hidden = torch.stack([out.hidden_states[layer] for layer in self.layers], 1).float().cpu()
                for row, i in enumerate(batch):
                    p, h = prompts[i], hidden[row]  # (L, T, D)
                    answer[i] = h[:, p.answer_position].half()
                    options[i] = torch.stack([h[:, a:b].mean(1) for a, b in p.option_spans]).half()
                    logprobs[i] = token_logprobs[row, p.option_codes]
                bar.update(len(batch))
        return answer, options, logprobs


def _mean_of_readings(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """The average of two float16 readings, computed in float32 and stored in float16."""
    return (0.5 * (first.float() + second.float())).half()


def _length_batches(prompts: Sequence[EncodedPrompt], budget: int) -> list[list[int]]:
    """Prompt indices grouped by length so that each padded batch stays within ``budget`` tokens."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].ids))
    batches, current, longest = [], [], 0
    for i in order:
        n = len(prompts[i].ids)
        if current and max(longest, n) * (len(current) + 1) > budget:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        batches.append(current)
    return batches
