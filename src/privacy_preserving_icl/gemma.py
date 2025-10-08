import dataclasses
import torch
from typing import Any, Sequence, AsyncGenerator, Protocol, Mapping, TypeAlias
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CompletionsSettings:
    temperature: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stop: str | Sequence[str] | None = None
    logprobs: int | None = None
    echo: bool | None = None
    logit_bias: Mapping[int, float] | None = None
    top_p: float | None = None


class TokenWithLogprob(Protocol):
    @property
    def text(self) -> str:
        """The textual representation of the token.

        For tokens that don't constitute valid UTF-8, this will look like "bytes:\\x??\\x??".
        """
        ...

    @property
    def bytes(self) -> bytes:
        "The bytes that make up the token."
        ...

    @property
    def logprob(self) -> float:
        ...

    @property
    def token_id(self) -> int:
        ...


@dataclass
class SampledToken:
    token: TokenWithLogprob
    top_choices: tuple[TokenWithLogprob, ...]


Completion: TypeAlias = Sequence[SampledToken]


@dataclass(frozen=True)
class GemmaTokenWithLogprob:
    text: str
    logprob: float
    token_id: int

    @property
    def bytes(self) -> bytes:
        return self.text.encode("utf-8")


class Encoder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.n_vocab = tokenizer.vocab_size

    def encode_single_token(self, character):
        # print(type(self.tokenizer(character).to(self.model.device)),  self.tokenizer(character).to(self.model.device))
        return tuple(self.tokenizer(character).to(self.model.device)['input_ids'])

    def encode(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)['input_ids']

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# @dataclass(frozen=True)
class Gemma:
    model_name_or_path: str
    default_completion_settings: dict[str, Any] = dataclasses.field(default_factory=dict)
    encoding: str

    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        self.model.eval()
        self.encoding = Encoder(self.tokenizer, self.model)

    def generate_next_token(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs, output_attentions=False, output_hidden_states=False)
            logits = outputs.logits[:, -1, :]  # logits for next token
            return logits

    async def completions(
            self,
            inputs: str | Sequence[int],
            settings: CompletionsSettings | None = None
    ) -> Sequence[Completion]:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = torch.tensor(prompt)
        output_settings = settings or CompletionsSettings()
        max_tokens = output_settings.max_tokens or 50

        with torch.no_grad():
            # outputs = self.model.generate(
            #     inputs,
            #     max_new_tokens=max_tokens,
            #     return_dict_in_generate=True,
            #     output_scores=True
            # )
            outputs = self.model(inputs)
            logits = outputs.logits[:, -1, :]  # Take logits for the last token
            probs = torch.softmax(logits, dim=-1)

        topk = torch.topk(probs, settings.logprobs)
        topk_ids = topk.indices.squeeze()
        topk_probs = topk.values.squeeze()

        return [[SampledToken(
            None,
            [GemmaTokenWithLogprob('', logprob, token_id.item()) for token_id, logprob in
             zip(topk_ids, topk_probs.tolist())]
            # Top-k logprobs not supported by default
        )]]

        # generated_tokens = outputs.sequences[0][inputs.shape[-1]:]
        # scores = outputs.scores[0]

        # print(scores)
        # tokens = []
        # for token_id, score  in enumerate(scores):
        #     # logprobs = torch.log_softmax(score, dim=-1)
        #     text = self.tokenizer.decode(token_id, skip_special_tokens=True)
        #     tokens.append(
        #         SampledToken(
        #             GemmaTokenWithLogprob(text, score, token_id),
        #             tuple()  # Top-k logprobs not supported by default
        #         )
        #     )
        # print('I am god', tokens)
        # return [tokens]

    def concat(self, tensor, lst):
        # print(input_ids, new_input_ids)
        # print(tensor, lst, "here I am")
        if lst:
            lst_tensor = torch.tensor([lst], device=self.model.device)
            return torch.cat((tensor, lst_tensor), dim=1)

        return tensor

    async def streaming_completions(
            self,
            prompt: str | Sequence[int],
            settings: CompletionsSettings | None = None
    ) -> Sequence[AsyncGenerator[SampledToken, None]]:
        # Simulate streaming by yielding one token at a time
        async def generator() -> AsyncGenerator[SampledToken, None]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_settings = settings or CompletionsSettings()
            max_tokens = output_settings.max_tokens or 50

            input_len = inputs['input_ids'].shape[-1]
            past_key_values = None
            input_ids = inputs['input_ids']

            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                probs = torch.nn.functional.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)
                logprobs = torch.log_softmax(logits, dim=-1)
                logprob = logprobs[0, token_id.item()].item()
                text = self.tokenizer.decode(token_id.item(), skip_special_tokens=True)

                yield SampledToken(
                    GemmaTokenWithLogprob(text, logprob, token_id.item()),
                    tuple()
                )

                input_ids = token_id.unsqueeze(0)

        return [generator()]
