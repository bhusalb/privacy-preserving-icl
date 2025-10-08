from collections.abc import Sequence, Set
from privacy_preserving_icl.gemma import Gemma, CompletionsSettings
from privacy_preserving_icl.prob_utils import log_normalize


def load_gemma_model(model='google/gemma-1.1-2b-it'):
    return Gemma(model)


MAX_TOP_LOGPROBS = 100
MAX_LOGIT_BIAS = 100
MIN_LOGIT_BIAS = -100


async def next_logprobs(
        self, prompt: str | Sequence[int], top_p=1
) -> dict[int, float]:
    [sampled_tokens] = await self.completions(
        prompt,
        CompletionsSettings(n=1, max_tokens=1, logprobs=100, stop=["<eos>"]),
    )
    if len(sampled_tokens) == 0:
        if isinstance(prompt, str):
            prompt += "<eos>"
        else:
            prompt = [*prompt, self.encoding.encode_single_token("<eos>")]
        [[*_prev_tokens, sampled_token]] = await self.completions(
            prompt,
            CompletionsSettings(
                n=1, max_tokens=0, logprobs=100, echo=True, top_p=top_p
            ),
        )
    else:
        [sampled_token] = sampled_tokens

    return {tlp.token_id: tlp.logprob for tlp in sampled_token.top_choices}


async def normalized_logprobs_for_chosen_tokens(
        self, prompt: Sequence[int], chosen_tokens: Set[int], top_p: float
) -> dict[int, float]:
    """Compute the probability that the prompt will be continued with each of the chosen tokens.

    The returned probability distribution is normalized over just the chosen tokens."""
    assert (
            len(chosen_tokens) <= MAX_TOP_LOGPROBS
    ), f"chosen_tokens must be <= {MAX_TOP_LOGPROBS} in length"

    logit_bias = {token_id: MAX_LOGIT_BIAS for token_id in chosen_tokens}
    [sampled_tokens] = await self.completions(
        prompt,
        CompletionsSettings(
            n=1,
            max_tokens=1,
            logprobs=MAX_TOP_LOGPROBS,
            logit_bias=logit_bias,
            top_p=top_p,
        ),
    )
    if len(sampled_tokens) == 0:
        # Fall back to querying over the set
        chosen_tokens_list = list(chosen_tokens)

        result = await self.completions(
            [[*prompt, token_id] for token_id in chosen_tokens_list],
            CompletionsSettings(n=1, max_tokens=0, logprobs=0, echo=True),
        )
        unnormalized_logprobs = {
            sampled_token.token.token_id: sampled_token.token.logprob
            for [*_prev_tokens, sampled_token] in result
        }
        return log_normalize(unnormalized_logprobs)
    else:
        [sampled_token] = sampled_tokens
        biased_logprobs = {
            tlp.token_id: tlp.logprob for tlp in sampled_token.top_choices
        }
        biased_logprobs_for_tokens = {
            token_id: biased_logprobs.get(token_id, float("-inf"))
            for token_id in chosen_tokens
        }
        return log_normalize(biased_logprobs_for_tokens)
