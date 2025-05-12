from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.hf import BltTokenizerAndPatcher
import torch.nn as nn
import logging
import os

import torch

entropy_repo = "facebook/blt-entropy"
blt_repo = "facebook/blt-1b"
entropy_model = LMTransformer.from_pretrained(entropy_repo)
model = ByteLatentTransformer.from_pretrained(blt_repo)
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(blt_repo)
tokenizer = tok_and_patcher.tokenizer_args.build()
patcher = tok_and_patcher.patcher_args.build()

device = torch.cuda.current_device()

# model = nn.DataParallel(model)
# entropy_model = nn.DataParallel(entropy_model)
# model.to(device)
# entropy_model.to(device)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
entropy_model = entropy_model.to("cuda" if torch.cuda.is_available() else "cpu")


from bytelatent.args import EvalArgs
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.patcher import Patcher
from bytelatent.distributed import (
    DistributedArgs,
    dist_max,
    dist_min,
    dist_sum,
    get_device_mesh,
    setup_torch_distributed,
)
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer

logger = logging.getLogger()


def get_max_length(input_tokens: list[list[int]] | None) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        max_length = 0
    else:
        max_length = max([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        max_length = int(dist_max(max_length))
    return max_length


def get_min_length(input_tokens: list[list[int]] | None) -> int:
    # reduce min length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        # TODO: Double check this change from int(1e9) is correct
        min_length = 0
    else:
        min_length = min([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        min_length = int(dist_min(min_length))
    return min_length


def get_generation_range(
    prompt_tokens: list[list[int]] | None, max_gen_len: int
) -> tuple[int, int]:
    batch_min_prompt_length = get_min_length(prompt_tokens)
    batch_max_prompt_length = get_max_length(prompt_tokens)
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate_nocache(
    prompts: list[str] | None,
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    patcher: Patcher,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
) -> list[list[int]]:
    assert (
        patcher.realtime_patching
    ), "generate_nocache requires patcher.realtime_patching=True"
    model.eval()
    if prompts is None:
        prompt_tokens = None
        n_truncated_prompts = 0
        total_truncated_prompts = 0
    else:
        prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
        n_truncated_prompts = sum([max_prompt_len < len(t) for t in prompt_tokens])
        total_truncated_prompts = 0

        # Truncation
        prompt_tokens = [
            t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
            for t in prompt_tokens
        ]

    if total_truncated_prompts > 0:
        logger.info(
            f"There are {total_truncated_prompts} prompts that are truncated on the left, "
            f"length greater than max_prompt_len = {max_prompt_len}, "
            f"maximum prompt length = {get_max_length(prompt_tokens)} across all gpus."
        )

    if prompt_tokens is None:
        prompt_tokens = [[tokenizer.bos_id] for _ in range(end_pos)]

    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    batch_size = len(prompt_tokens)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id).cuda().long()

    # Copy inputs to tensor for generated tokens
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long()
    input_text_mask = tokens != tokenizer.pad_id

    for i, curr_pos in enumerate(range(start_pos, end_pos)):
        current_tokens = tokens[:, :curr_pos]
        patch_lengths, _ = patcher.patch(current_tokens, include_next_token=True)
        logits = model(current_tokens, patch_lengths=patch_lengths)[:, -1]

    return logits


def launch_generate(eval_args: EvalArgs):
    

    # fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile)
    # if (
    #     fs.exists(eval_args.ckpt_dir)
    #     and fs.exists(os.path.join(eval_args.ckpt_dir, "params.json"))
    #     and len(fs.glob(os.path.join(eval_args.ckpt_dir, "*.pth"))) != 0
    # ):
    #     consolidate_path = eval_args.ckpt_dir
    # else:
    #     raise ValueError("Did not find a consolidated checkpoint in the ckpt_dir")

    # model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
    #     consolidate_path,
    # )
    # patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher.realtime_patching = True
    patcher.entropy_model = entropy_model
    #patcher.entropy_model_checkpoint_dir = '/mnt/sdrive/yasod/blt/hf-weights/blt-1b/entropy_model'
    # patcher = patcher_args.build()
    outputs = generate_nocache(
        eval_args.prompts, model=model, tokenizer=tokenizer, patcher=patcher
    )
    print(outputs.shape)
    sys.exit()
    text_outputs = [tokenizer.decode(t) for t in outputs]
    for p, t in zip(eval_args.prompts, text_outputs):
        print(f'Prompt: "{p}" Completion: "{t}"')
        print()


eval_args = parse_args_to_pydantic_model(EvalArgs)
traffic = ['5a70_7088_88ff_ff9f_9f75_7504_04d9_d9f5', '5a70_7ab8_dfff_ffkdf_9975_1504_04e9_d2f5']
eval_args.prompts = traffic
launch_generate(eval_args)

