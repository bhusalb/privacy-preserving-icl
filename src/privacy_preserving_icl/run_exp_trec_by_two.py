import asyncio
import math
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

from privacy_preserving_icl.lm import load_gemma_model


DEFAULT_NUM_PRIVATE_TRAIN = 80
DEFAULT_NUM_PUBLIC_TRAIN = 0
DEFAULT_NUM_VALID = 4
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 40
DEFAULT_NUM_TEST = 1000

labels = ["Ab", "Entity", "Description", "Person", "Location", "Number"]
label_dict = {
    0: ["Ab"],
    1: ["Entity"],
    2: ["Description"],
    3: ["Person"],
    4: ["Location"],
    5: ["Number"],
}

def format_full_datum_for_prompt(labels, datum: dict[str, str]):
    return f'Text: "{datum["text"]}"\n'


def construct_prompt_same(train_examples, test_example):
    prompt = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
    for train_example in train_examples:
        prompt += "Question: " + train_example["text"] + "\n"
        prompt += (
            "Answer Type: " + label_dict[train_example["coarse_label"]][0] + "\n\n"
        )
    prompt += "Question: " + test_example["text"] + "\n"
    prompt += "Answer Type:"
    return prompt


def complete(prompts, l, model_name=None, temp=0.0, num_log_probs=1, echo=False, lm=None):
    """
    Simulates the OpenAI API `Completion.create()` function using a local HuggingFace model.

    Args:
        prompts (str or list of str): Input prompt(s)
        l (int): Number of tokens to generate (0 = log-prob of last token, 1 = predict next token)
        model_name: Ignored; kept for compatibility
        temp (float): Temperature (0 = greedy)
        num_log_probs (int): How many top log-probs to return (used if l == 1)
        echo (bool): Whether to include the prompt in output text (used if l > 1)
        lm (tuple): (model, tokenizer)

    Returns:
        dict: {"choices": [ { "logprobs": { ... } }, ... ]}
    """

    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]

    results = []

    for prompt in prompts:
        inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)

        if l == 0:
            # Compute log-prob of the last token in the prompt
            with torch.no_grad():
                outputs = lm.model(**inputs)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]
                log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
                last_token_id = inputs.input_ids[0, -1].item()
                # Use second-to-last token position's logits to get log-prob of last token
                token_logprob = log_probs[-1, last_token_id].item()

            result = {
                "logprobs": {
                    "token_logprobs": [token_logprob]
                }
            }

        else:
            # Predict the next token and return top `num_log_probs`
            with torch.no_grad():
                outputs = lm.model(**inputs)
                logits = outputs.logits[:, -1, :]  # logits for next token
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, k=num_log_probs, dim=-1)
            top_values = topk.values[0].cpu().numpy()
            top_indices = topk.indices[0].cpu().numpy()

            top_logprobs = {}
            for i in range(num_log_probs):
                token_str = lm.tokenizer.decode([top_indices[i]])
                top_logprobs[token_str] = float(top_values[i])

            result = {
                "logprobs": {
                    "top_logprobs": [top_logprobs]
                }
            }

        results.append(result)
    
    return { "choices": results }


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_model_response(data, test_examples, openai_model, lm):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example))

    chunked_prompts = list(chunks(prompts, 20))
    for test_chunk in chunked_prompts:
        response = complete(test_chunk, l=1, model_name=openai_model, num_log_probs=100, lm=lm)

        for answer_id, answer in enumerate(response["choices"]):
            all_raw_answers.append(answer)

    return all_raw_answers


def get_label_probs(all_raw_answers, test_subset):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(label_dict)
    approx = False
    assert len(all_raw_answers) == len(test_subset)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    cnt = 0
    for i, ans in enumerate(all_raw_answers):
        try:
            top_logprobs = ans["logprobs"]["top_logprobs"][
                0
            ]  # [0] since we only ask for complete one more token
        except:
            cnt += 1  # cnt for corner case
        label_probs = [0] * len(label_dict.keys())
        for j, label_list in label_dict.items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j)  # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)  # prob not normalized

    return all_label_probs  # NOT NORMALIZED


def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        if np.sum(label_probs) > 0:  # corner case np.sum(label_probs)=0.
            label_probs = label_probs / np.sum(label_probs)  # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def get_p_content_free(train_subset, openai_model, content_free_inputs=("N/A",), lm=None):
    """Query model with content free input, return its prediction probability for each label"""
    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt_same(train_subset, content_free_input)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(
                    complete(
                        prompt + " " + a, 0, openai_model, echo=True, num_log_probs=1, lm=lm
                    )["choices"][0]["logprobs"]["token_logprobs"][-1]
                )
            p_y[i] = prob
        all_p_y.append(p_y)
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y


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
    labels,
    gen_seed,
    max_tokens,
    clipping_bound
) -> list[int]:
    generated_token_ids: list[int] = []

    public_prompt = public_prompt_template.format(label=labels[label])
    public_prompt_tokens = lm.encoding.encode(public_prompt)



    assert num_private_train_splits > 0
    select_list = []
    for i in range(len(trainset)):
        if trainset[i]["coarse_label"] == label:
            select_list.append(i)
    train_subset = trainset.select(select_list, keep_in_memory=True)
   
    
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
        prompt_template.format(label=labels[label], text="\n".join(format_full_datum_for_prompt(labels, datum) for datum in split))
        for split in private_train_splits
    ]

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


def select_uniform_n_shots_over_labels(data, n_shots):
    select_list = []
    n_shots_per_label = math.ceil(n_shots / len(labels))
    labels_counter = {label[1][0]: n_shots_per_label for label in label_dict.items()}
    n_shots_selected = 0
    for i in range(len(data)):
        label = label_dict[data[i]["coarse_label"]][0]
        if labels_counter[label] == 0 or data[i]["coarse_label"] == 0:
            continue
        else:
            labels_counter[label] -= 1
            select_list.append(i)
            n_shots_selected += 1
        if n_shots_selected == n_shots:
            break
    query_subset = data.select(select_list, keep_in_memory=True)
    return query_subset


def _main(
    temperature: Annotated[float, typer.Option()],  # noise parameters
    num_private_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PRIVATE_TRAIN,
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid: Annotated[int, typer.Option()] = DEFAULT_NUM_VALID,
    # num_private_train_splits=M
    num_private_train_splits: Annotated[
        int, typer.Option()
    ] = DEFAULT_NUM_PRIVATE_TRAIN_SPLITS,
    num_test: Annotated[int, typer.Option()] = DEFAULT_NUM_TEST,
    # no_public_token=True, RVP=False; no_public_token=False, RVP=True
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 100,
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
    clipping_bound: Annotated[float, typer.Option()] = 10,
    filename: Annotated[str, typer.Option()] = "trec.jsonl",
):
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
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

        data = cast(DatasetDict, load_dataset("trec", 'default', trust_remote_code=True))
        print(labels)

        trainset = data["train"].shuffle(seed=synth_seed, keep_in_memory=True)
        print("trainset length", len(trainset))
      

        trainset = trainset.select(
            range(len(trainset)), keep_in_memory=True
        )
        queryset = data["train"].shuffle(seed=eval_seed, keep_in_memory=True)
        query_subset = select_uniform_n_shots_over_labels(queryset, num_valid)

        
        synthetic_examples = []

        # Turn the data into prompts
        # instruction = "Given a label of document type, generate the chosen type of document accordingly.\n\n"

        stop_tokens_ids = {lm.tokenizer.eos_token_id, 108, 109, 110,111, 112, 113, 114, 115, 116, 117, 118, 119, 120}

                    # Turn the data into prompts
        prompt_template = """# [User] 
Here are questions with Answer Type: {label}.

{text}

Please give me another one.

# [Assistant]
Question:
"""
        public_prompt_template = """#[User]
Generate only a question with Answer Type {label}.

#[Assistant]
Question:"""

        
        for i, test_datum in enumerate(query_subset, 1):
            print(f"# Example {i}")
            print(f'Answer Type: "{labels[test_datum["coarse_label"]]}"')
            print(f'References:\n "{test_datum["text"]}"')


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
                test_datum["coarse_label"],
                lm,
                noise_rng,
                temperature,
                labels,
                gen_seed,
                max_token_per_text- 1,  # need one token length for EOS.
                clipping_bound
            )

            generated = lm.encoding.decode(generated_token_ids).rstrip('"')

            print(f"Generated: {generated}\n")
            output_datum = {}
            output_datum["text"] = generated.strip()
            output_datum["coarse_label"] = test_datum["coarse_label"]
            synthetic_examples.append(output_datum)

        test_subset = data["test"]
        test_labels = [test_example["coarse_label"] for test_example in test_subset]

        content_free_inputs = [{"text": "N/A"}, {"text": ""}, {"text": "[MASK]"}]
        p_cf_wout_DP = get_p_content_free(
            query_subset, None, lm=lm, content_free_inputs=content_free_inputs
        )

        all_raw_answers_wout_DP = get_model_response(
            query_subset, test_subset, None, lm=lm
        )
        all_label_probs_wout_DP = get_label_probs(
            all_raw_answers_wout_DP, test_subset
        )

        acc_original_wout_DP = eval_accuracy(all_label_probs_wout_DP, test_labels)
        acc_calibrated_wout_DP = eval_accuracy(
            all_label_probs_wout_DP,
            test_labels,
            mode="diagonal_W",
            p_cf=p_cf_wout_DP,
        )

        print(f"Accuracy (original) without DP: {acc_original_wout_DP}")
        print(f"Accuracy (calibrated) without DP: {acc_calibrated_wout_DP}")

        p_cf_w_DP = get_p_content_free(
            synthetic_examples,
            None,
            content_free_inputs=content_free_inputs,
            lm=lm
        )

        all_raw_answers_w_DP = get_model_response(
            synthetic_examples, test_subset, None, lm=lm
        )
        all_label_probs_w_DP = get_label_probs(
            all_raw_answers_w_DP, test_subset
        )

        acc_original_w_DP = eval_accuracy(all_label_probs_w_DP, test_labels)
        acc_calibrated_w_DP = eval_accuracy(
            all_label_probs_w_DP, test_labels, mode="diagonal_W", p_cf=p_cf_w_DP
        )

        print(f"Accuracy (original) with DP: {acc_original_w_DP}")
        print(f"Accuracy (calibrated) with DP: {acc_calibrated_w_DP}")

        final_output = {
            'accuracy_without_dp': acc_original_wout_DP,
            'accuracy_without_dp_cali': acc_calibrated_wout_DP,
            'db': 'trec',
            'type': 'our',
            'nshots': num_valid,
            'synthetic_examples': synthetic_examples,
            'seed': synth_seed,
        }

        print(f"Accuracy (original) with DP: {acc_original_w_DP}")
        print(f"Accuracy (calibrated) with DP: {acc_calibrated_w_DP}")

        final_output['accuracy_with_dp'] = acc_original_w_DP
        final_output['accuracy_with_dp_cali'] = acc_calibrated_w_DP
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
