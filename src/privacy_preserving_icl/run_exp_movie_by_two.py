import asyncio
import re
import sys
import traceback
from collections.abc import Iterable
from typing import Annotated, cast
import torch
import more_itertools
import numpy as np
import tqdm
import typer
from datasets import DatasetDict, load_dataset
import torch.nn.functional as F
import json
from transformers import  StoppingCriteria,StoppingCriteriaList

from privacy_preserving_icl.lm import load_gemma_model


DEFAULT_NUM_PRIVATE_TRAIN = 80
DEFAULT_NUM_PUBLIC_TRAIN = 0
DEFAULT_NUM_VALID = 4
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 20
DEFAULT_NUM_TEST = -1


def format_full_datum_for_prompt(field_name, datum: dict[str, str]):
    return (
        f'Phrase: "{datum["label"]}"\nText: "{datum["content"]}"\n'
    )


def construct_prompt_same(train_examples, test_example, field_name):
    prompt = f""
    for train_example in train_examples:
        prompt += "Text: " + train_example["content"] + "\n"
        prompt += f"{field_name}: " + train_example["label"] + "\n\n"
    prompt += "Text: " + test_example["content"] + "\n"
    prompt += f"{field_name}:"
    return prompt




class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, start_len):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.stop_token_ids = {tokenizer.eos_token_id, 108, 109, 110,111, 112, 113, 114, 115, 116, 117, 118, 119, 120}

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only newly generated tokens
        new_tokens = input_ids[0, self.start_len:]
        
        for new_token in new_tokens:
            if new_token.item() in self.stop_token_ids:
                return torch.tensor([True], dtype=torch.bool, device=input_ids.device)
        
        return torch.tensor([False], dtype=torch.bool, device=input_ids.device)
# def complete(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
#     # call GPT-3 API until result is provided and then return it
#     response = None
#     received = False
#     while not received:
#         try:
#             response = openai.Completion.create(
#                 engine=model_name,
#                 prompt=prompt,
#                 max_tokens=l,
#                 temperature=temp,
#                 logprobs=num_log_probs,
#                 echo=echo,
#                 stop="\n",
#                 n=n,
#             )
#             received = True
#         except:
#             error = sys.exc_info()[0]
#             if (
#                 error == openai.error.InvalidRequestError
#             ):  # something is wrong: e.g. prompt too long
#                 print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
#                 assert False

#             print("API error:", error)
#             time.sleep(1)
#     return response

def complete(prompts, l, model_name, temp=0, num_log_probs=None, echo=False, n=None, lm=None):
    answers = []
    for prompt in prompts:
        inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
        start_len = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList([StopOnNewline(lm.tokenizer, start_len)])

        outputs = lm.model.generate(
            **inputs,
            max_new_tokens=l,
            stopping_criteria=stopping_criteria,
            cache_implementation="static"
        )

        

        generated_text = lm.tokenizer.decode(outputs[0][start_len:], skip_special_tokens=True)

        answers.append(generated_text)

    return {
        "choices": answers,
    }

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_model_response(
    data,
    test_examples,
    openai_model,
    field_name,
    max_token_to_fill=20,
    additional_tokens=None,
    lm=None,
):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example, field_name))

    if additional_tokens is not None:
        assert len(additional_tokens) == len(prompts)
        for i in range(len(prompts)):
            prompts[i] += additional_tokens[i]

    chunked_prompts = list(chunks(prompts, 20))
    for test_chunk in chunked_prompts:
        response = complete(
            test_chunk, l=max_token_to_fill, model_name=openai_model, num_log_probs=100, lm=lm
        )

        for answer_id, answer in enumerate(response["choices"]):
            all_raw_answers.append(answer)

    return all_raw_answers


def em_accuracy_helper(prediction, label):
    correctness_list = []
    for pred, l in zip(prediction, label):
        pred = pred.split("\n")[0]
        if pred == l:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def clip_logits(logits: torch.Tensor, c: float) -> torch.Tensor:
    """
    Clip and re-center logits using Equation (1).
    
    Args:
        logits: Input logit vector
        c: Clipping bound
        
    Returns:
        Clipped and re-centered logits
    """
    # # Find the maximum logit along the last dimension (row-wise max)
    # max_logit = torch.max(logits, dim=-1, keepdim=True).values
    # # Apply clipping: max{-c, zi - max_j{zj} + c}
    # shifted_logits = logits - max_logit + c
    # clipped_logits = torch.clamp(shifted_logits, min=-c, max=c)
    # return clipped_logits

    if c <= 0:
        raise ValueError("Clipping bound c must be positive.")

    # Subtract max for numerical stability, then add c
    max_logit = logits.amax(dim=-1, keepdim=True)
    shifted = logits - max_logit + c
    # Clamp to [-c, c]
    clipped = torch.clamp(shifted, min=-c, max=c)
    # Optional: ensure no NaN or Inf
    if torch.isnan(clipped).any() or torch.isinf(clipped).any():
        raise ValueError("Clipped logits contain NaN or Inf values.")
    return clipped


async def generate_with_private_prompts(
    trainset,
    num_private_train,
    num_private_train_splits,
    prompt_template,
    public_prompt_template,
    stop_tokens_ids,
    label,
    lm,
    noise_rng,
    temperature,
    field_name,
    gen_seed,
    max_tokens,
    clipping_bound
) -> list[int]:
    
    generated_token_ids: list[int] = []

    public_prompt = public_prompt_template.format(
        field_name=field_name, keyword=label
    )
    public_prompt_tokens = lm.encoding.encode(public_prompt)

    train_subset = trainset.select(range(len(trainset)), keep_in_memory=True)


    private_train_subset = cast(
        Iterable[dict[str, str]],
        train_subset.shuffle(gen_seed, keep_in_memory=True).select(
            range(num_private_train), keep_in_memory=True
        ),
    )
    private_train_splits = [
        list(it)
        for it in more_itertools.distribute(
            num_private_train_splits, private_train_subset
        )
    ]
    private_prompts = [
        prompt_template.format(field_name=field_name, text="\n".join(format_full_datum_for_prompt(field_name, datum) for datum in split),  keyword=label)
        for split in private_train_splits
    ]

    print(public_prompt, private_prompts[0])

    private_prompts_tokens = [
        lm.encoding.encode(prompt) for prompt in private_prompts
    ]

    # print("Private prompts:", private_prompts)
    for _ in tqdm.tqdm(range(max_tokens), total=float("inf"), unit=" tokens generated"):   
        mean_logits = None
        for idx, prompt in enumerate(private_prompts_tokens):
            next_logprobs = lm.generate_next_token(lm.concat(prompt, generated_token_ids))
            clipped = clip_logits(next_logprobs, clipping_bound)
            if mean_logits is None:
                mean_logits = clipped.clone()
            else:
                # Running mean: mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                mean_logits += (clipped - mean_logits) / (idx + 1)
        Z_bar = mean_logits

        # private_next_logprobs = torch.stack(private_next_logprobs) # 200 , 1 , 256000
        public_logits = lm.generate_next_token(lm.concat(public_prompt_tokens, generated_token_ids))
        # Z_bar = clip_logits(private_next_logprobs, clipping_bound).mean(dim=0)
        # del private_next_logprobs
        # Z_bar = torch.stack([clip_logits(logits, clipping_bound) for logits in private_next_logprobs]).mean(dim=0).to(lm.model.device)
        U = clip_logits(public_logits, clipping_bound)
        
        Z_mix = (Z_bar + U) / 2
        probs = F.softmax(Z_mix / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        # print(f"Next token ID: {next_token_id}, decoded: {lm.encoding.decode([next_token_id])}")
        # print(f"Generated token ID: {next_token_id}, decoded: {lm.encoding.decode([next_token_id])}")
        if next_token_id in stop_tokens_ids or next_token_id == lm.tokenizer.eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        # Explicitly delete tensors and clear cache
        del public_logits, Z_bar, U, Z_mix, probs

    return generated_token_ids





def _main(
    temperature: Annotated[float, typer.Option()],  # noise parameters
    # num_private_train=MN. MN=0 with num_valid=4 will get epsilon=0 (4-shot) results.
    num_private_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PRIVATE_TRAIN,
    # by default set to 0. set_num_public_train >0 indicates additional public data available.
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid: Annotated[int, typer.Option()] = DEFAULT_NUM_VALID,
    # num_private_train_splits=M
    num_private_train_splits: Annotated[
        int, typer.Option()
    ] = DEFAULT_NUM_PRIVATE_TRAIN_SPLITS,
    num_test: Annotated[int, typer.Option()] = DEFAULT_NUM_TEST,
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 20,
    # random seed for subsampling in generation
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
    # choice bewteen ["Genre", "Director"]
    field_name: Annotated[str, typer.Option()] = "Genre",
    data_path: Annotated[str, typer.Option()] = "./data/movie",
    clipping_bound: Annotated[float, typer.Option()] = 10,
    filename: Annotated[str, typer.Option()] = "movie.jsonl",
):
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
        assert field_name in [
            "Director",
            "Genre",
        ]  # field_name from movie dataset include "Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith("--"):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match("^[a-zA-Z0-9_]+$", x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = " ".join(command)
        print(command)


        lm = load_gemma_model()
        noise_rng = np.random.RandomState()

        data_files = {"train": "train.csv", "test": "test.csv"}
        data = cast(
        DatasetDict,
        load_dataset(f"{data_path}/{field_name}", data_files=data_files),
        )

        trainset = data["train"].shuffle(seed=synth_seed, keep_in_memory=True)
        print("trainset length", len(trainset))

        trainset = trainset.select(
            range(len(trainset)), keep_in_memory=True
        )
        query_subset = (
        data["train"]
            .shuffle(seed=eval_seed, keep_in_memory=True)
            .select(range(num_valid), keep_in_memory=True)
        )


        synthetic_examples = []

        prompt_template = '''# [User] 
Give me text about a film and the extracted Phrase about its {field_name}.
{text}

Please give me another Phrase and Text. IMPORTANT: The exact {field_name} phrase "{keyword}" must be mentioned in Text.

# [Assistant]
Phrase: "{keyword}"
Text: "'''
        public_prompt_template = '''#[User]
Give me text about a film and the extracted Phrase about its {field_name}. IMPORTANT: The exact {field_name} phrase "{keyword}" must be mentioned in Text.

# [Assistant]
Phrase: "{keyword}"
Text: "'''

        stop_tokens_ids = {lm.tokenizer.eos_token_id, 108, 109, 110,111, 112, 113, 114, 115, 116, 117, 118, 119, 120}

            
        for i, test_datum in enumerate(query_subset, 1):
            print(f"# Example {i}")
            print(f'{field_name}: "{test_datum["label"]}"')

            np.random.seed(synth_seed + i)
            gen_seed = np.random.randint(100000)
            print(f"gen-seed: {gen_seed}")

            generated_token_ids = await generate_with_private_prompts(
                trainset,
                num_private_train,
                num_private_train_splits,
                prompt_template,
                public_prompt_template,
                stop_tokens_ids,
                test_datum["label"],
                lm,
                noise_rng,
                temperature,
                field_name,
                gen_seed,
                max_token_per_text- 1,
                clipping_bound
            )
        

            generated = lm.encoding.decode(generated_token_ids).rstrip('"')

            print(f"Generated: {generated}\n")
            output_datum = {}
            output_datum["content"] = generated.strip()
            output_datum["label"] = test_datum["label"]
            synthetic_examples.append(output_datum)
  
    
        if num_test > 0 and num_test <= len(data["test"]):
            test_subset = (
                data["test"]
                .shuffle(seed=12345, keep_in_memory=True)
                .select(range(num_test), keep_in_memory=True)
            )
        else:
            test_subset = data["test"]

        all_raw_answers_wout_DP = get_model_response(
            query_subset, test_subset, None, field_name,  lm=lm
        )
        all_orig_ans = []
        # for resp in all_raw_answers_wout_DP:
            # all_orig_ans.append(resp["text"])

        for resp in all_raw_answers_wout_DP:
            # abc = list(resp['logprobs']['top_logprobs'][0].keys())
            # all_orig_ans.append(abc[0].lower().strip())  # take the first token as the answer
            all_orig_ans.append(resp.strip())  # take the first token as the answer

        # all_orig_ans = [ans.strip() for ans in all_orig_ans]
        test_labels = test_subset["label"]
        orig_accuracy = em_accuracy_helper(all_orig_ans, test_labels)
        print(f"Accuracy (original) without DP: {orig_accuracy}")

        
        all_raw_answers_w_DP = get_model_response(
            synthetic_examples, test_subset, None, field_name, lm=lm
        )
        all_orig_ans = []
        # for resp in all_raw_answers_w_DP:
        #     all_orig_ans.append(resp["text"])

        for resp in all_raw_answers_w_DP:
            # abc = list(resp['logprobs']['top_logprobs'][0].keys())
            # all_orig_ans.append(abc[0].lower().strip())  # take the first token as the answer
            all_orig_ans.append(resp.strip())  # take the first token as the answer

        # all_orig_ans = [ans.strip() for ans in all_orig_ans]
        test_labels = test_subset["label"]
        orig_accuracy_dp = em_accuracy_helper(all_orig_ans, test_labels)
        print(f"Accuracy (original) with DP: {orig_accuracy_dp}")


        final_output = {
            'accuracy_without_dp': orig_accuracy,
            'accuracy_without_dp_cali': orig_accuracy,
            'accuracy_with_dp': orig_accuracy_dp,
            'accuracy_with_dp_cali': orig_accuracy_dp,
            'db': 'movie',
            'field_name': field_name,
            'type': 'our',
            'nshots': num_valid,
            'synthetic_examples': synthetic_examples,
            'seed': synth_seed,
        }

        print('\n\n')
        print(json.dumps(final_output))
        with open(filename, "a") as f:
            f.write(json.dumps(final_output) + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    typer.run(_main)
