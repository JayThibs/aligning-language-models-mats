import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, GPTJForCausalLM
from time import time
import pandas as pd
import os


def gpt_generate(
    text="Hello, world!",
    model_name="EleutherAI/gpt-j-6B",
    model=None,
    tokenizer=None,
    temperature=0.2,
    txt_path=None,
    stop_token="\n",
    stop_completion_on_token=False,
    num_return_sequences=1,
    gpu=False,
    with_log_probs=False,
    max_length=50,
    min_length=1,
    no_outputs=False,
    time_test=False,
    save_completions=False,
    only_print_completions=False,
    no_prints=False,
):

    if gpu:
        device_str = "GPU"
        device = torch.device("cuda")
    else:
        device_str = "CPU"
        device = torch.device("cpu")

    if not time_test:
        if no_prints:
            print(f"Using device: {device}.")

    if txt_path:
        with open(txt_path, "r") as f:
            text = f.read()

    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
    if tokenizer == model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    min_length = len(input_ids[0]) + 5
    max_length = max_length + len(input_ids[0])

    if stop_completion_on_token:
        stop_words = ["\n"]
        stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteria(stop_ids)

    start = time()
    generated_outputs = model.generate(
        input_ids,
        do_sample=True,
        early_stopping=True,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
        device=device,
        repetition_penalty=1.2,
        length_penalty=0.8,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop_criteria])
        if stop_completion_on_token
        else StoppingCriteriaList(),
        temperature=temperature,
    )
    end = time()

    # # only use id's that were generated
    # # gen_sequences has shape [3, 15]
    # gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]

    # # let's stack the logits generated at each step to a tensor and transform
    # # logits to probs
    # probs = torch.stack(generated_outputs.scores, dim=1).softmax(
    #     -1
    # )  # -> shape [3, 15, vocab_size]

    # # now we need to collect the probability of the generated token
    # # we need to add a dummy dim in the end to make gather work
    # gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    # print(gen_probs)
    # # print(unique_normed_prob_per_sequence)

    # # now we can do all kinds of things with the probs

    # # 1) the probs that exactly those sequences are generated again
    # # those are normally going to be very small
    # unique_prob_per_sequence = gen_probs.prod(-1)

    # # 2) normalize the probs over the three sequences
    # normed_gen_probs = gen_probs / gen_probs.sum(0)
    # assert (
    #     normed_gen_probs[:, 0].sum() == 1.0
    # ), "probs should be normalized, rerun in case it's a floating point error"

    # # 3) compare normalized probs to each other like in 1)
    # unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)

    # all_log_probs = torch.stack(generated_outputs.scores, dim=1)
    # print(all_log_probs)
    # log_probs = torch.gather(all_log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    # print(log_probs)
    # mean_log_probs = torch.mean(log_probs)
    # print(mean_log_probs)

    if time_test:
        return end - start

    if no_prints:
        print("-----------------------------------------------------")
        print(
            f"Generated {num_return_sequences} sequences in {end-start:.2f} seconds with a {device_str}."
        )
        print("-----------------------------------------------------")

    if not no_outputs:
        print("~~~ Generated completion(s): ~~~ \n")
        if save_completions:
            saved_completions_path = "data/saved_completions.csv"
            saved_completions = []
        for i, sequence in enumerate(generated_outputs.sequences):
            if with_log_probs:
                token_list = []
                for token in sequence:
                    token_list.append(tokenizer.decode(token))
            generated_text = tokenizer.decode(sequence)
            generated_text = generated_text.replace("<|endoftext|>", "")
            if save_completions:
                saved_completions.append(generated_text)
            if only_print_completions:
                generated_text = " ".join(generated_text.split("relevant because")[1:])
            if not no_prints:
                print(f"Generation {i+1}. {generated_text}")
            # print(".".join(generated_text.split(".")[0:-2]) + ".")

            if with_log_probs:
                gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]
                # print(gen_sequences)
                # print(gen_sequences[i])
                print("----------------------------------------------------")
                print("Here are the log probabilities of the generated tokens:")
                all_log_probs = torch.stack(generated_outputs.scores, dim=1)
                log_probs = torch.gather(
                    all_log_probs, 2, gen_sequences[:, :, None]
                ).squeeze(-1)[i]
                token_with_log_probs = [
                    token_list[len(input_ids[0]) :],
                    log_probs.cpu().numpy(),
                ]
                df = pd.DataFrame(token_with_log_probs).T
                print(df)
                print("----------------------------------------------------")

        if save_completions:
            # tmp_df = pd.DataFrame(
            #     {"completions": saved_completions, "pass/fail": "fail"}
            # )
            # if os.path.exists(saved_completions_path):
            #     completions_df = pd.read_csv(saved_completions_path)
            #     completions_df.concat(tmp_df)
            # else:
            #     completions_df = tmp_df
            # completions_df.to_csv("data/saved_completions.csv", index=False)

            return saved_completions


def create_prompt_txt_from_df(
    df,
    row_idx,
    prompt_path,
    context_path="prompts/contexts/users_on_website.txt",
    task_description_path="prompts/task_description/task_description_1.txt",
    template_path="prompt_qa_template.txt",
    print_prompt=False,
) -> str:
    """
    Takes in dataframe of the dataset and creates a .txt of the prompt for row_index.
    """
    question = df["question"].iloc[row_idx]
    answer = df["answer"].iloc[row_idx]
    relevance = df["relevance"].iloc[row_idx]
    with open(context_path, "r") as f:
        context = f.read()

    with open(task_description_path, "r") as f:
        task_description = f.read()

    with open(template_path) as f:
        content = f.read()
        content = (
            content.replace("<<CONTEXT>>", context)
            .replace("<<QUESTION>>", question)
            .replace("<<ANSWER>>", answer)
            .replace("<<RELEVANCE>>", relevance)
            .replace("<<TASK DESCRIPTION>>", task_description)
        )
        with open(prompt_path, "w") as f:
            f.write(content)
            if print_prompt:
                print(content)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


def print_ground_truth_from_df(df, row_idx, prompt_path):
    create_prompt_txt_from_df(df, row_idx, prompt_path)
    with open(prompt_path, "r") as f:
        prompt = f.read()
    prompt_with_explanation = prompt + " " + df["explanation"].iloc[row_idx]
    with open(prompt_path, "w") as f:
        f.write(prompt_with_explanation)
    print(prompt_with_explanation)
