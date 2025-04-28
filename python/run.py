import argparse
import os
from typing import List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training lora on a custom dataset.")
    parser.add_argument(
        "--directory", "--dir", "-d",
        type=str,
        required=True,
        help="Path to the working directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--resume-checkpoint", "--resume", "-r",
        type=int,
        required=True,
        help="Resume training from a checkpoint."
    )
    parser.add_argument(
        "--batch-size", "--bs",
        type=str,
        required=True,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--gradient-accumulation-steps", "--grad",
        type=int,
        default=1,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--dataset-shuffle-buffer-size", "--shuffle",
        type=int,
        default=64,
        help="Shuffle buffer size for the dataset."
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-4,
        help="Batch size for the dataset."
    )
    parser.add_argument(
        "--checkpoint-interval", "--ci",
        type=int,
        default=251,
        help="Checkpoint interval."
    )
    parser.add_argument(
        "--validation-interval", "--vi",
        type=int,
        default=253,
        help="Validation interval."
    )
    parser.add_argument(
        "--cache-dir", "--cache",
        type=str,
        default=".cache",
        help="Cache directory."
    )
    parser.add_argument(
        "--precomputation-items", "--items",
        type=int,
        default=200,
        help="Number of items to precompute."
    )
    parser.add_argument(
        "--precomputation-once", "--once",
        action="store_true",
        help="Precompute items once."
    )
    parser.add_argument(
        "--training-steps", "-n",
        type=int,
        default=1000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--max-checkpoints", "--ckpt",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run in online mode."
    )
    parser.add_argument(
        "--tp-size", "--tp",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size", "--dp",
        type=int,
        default=1,
        help="Data parallel size."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full parameter training instead of lora."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    directory: str                   = args.directory
    model: str                       = args.model
    resume_checkpoint: int           = args.resume_checkpoint
    batch_size: str                  = args.batch_size
    gradient_accumulation_steps: int = args.gradient_accumulation_steps
    dataset_shuffle_buffer_size: int = args.dataset_shuffle_buffer_size
    learning_rate: float             = args.learning_rate 
    checkpoint_interval: int         = args.checkpoint_interval
    validation_interval: int         = args.validation_interval
    cache_dir: str                   = args.cache_dir
    precomputation_items: int        = args.precomputation_items
    precomputation_once: bool        = args.precomputation_once
    training_steps: int              = args.training_steps
    max_checkpoints: int             = args.max_checkpoints
    online_mode: bool                = args.online
    tp_size: int                     = args.tp_size
    dp_size: int                     = args.dp_size
    full: bool                       = args.full
    num_gpus: int                    = dp_size * tp_size

    cache_path = f"{directory}/{cache_dir}"
    os.makedirs(cache_path, exist_ok=True)

    default_training_json = f"training.json"
    default_validation_json = f"validation.json"

    if resume_checkpoint < 0:
        # find latest checkpoint folder
        try_resume_list: List[int] = []
        for folder in os.listdir(cache_path):
            if folder.startswith("finetrainers_step_"):
                assert os.path.isdir(os.path.join(cache_path, folder))
                try_resume_list.append(int(folder.split("_")[-1]))
        if not try_resume_list:
            print("No checkpoint found, starting from scratch.")
            resume_checkpoint = 0
        else:
            resume_checkpoint = max(try_resume_list)
            print(f"Resuming from checkpoint {resume_checkpoint}.")

    training_steps += resume_checkpoint # exclude the resume checkpoint from the training steps
    max_checkpoints = max(2, max_checkpoints) # ensure at least 2 checkpoints are kept

    parallel_cmd = [
        "--parallel_backend ptd",
        "--pp_degree 1",
        f"--dp_degree {dp_size}",
        "--dp_shards 1",
        "--cp_degree 1",
        f"--tp_degree {tp_size}",
    ]
    model_cmd = [
        "--model_name flux",
        f"--pretrained_model_name_or_path {model}",
    ]
    dataset_cmd = [
        f"--dataset_config {default_training_json}",
        f"--dataset_shuffle_buffer_size {dataset_shuffle_buffer_size}",
        "--enable_precomputation",
        f"--precomputation_items {precomputation_items}",
    ]
    if precomputation_once:
        dataset_cmd.append("--precomputation_once")
    training_cmd = [
        "--training_type lora",
        "--seed 42",
        f"--batch_size {batch_size}",
        f"--train_steps {training_steps}",
        "--rank 32",
        "--lora_alpha 32",
        "--target_modules \"transformer_blocks.*(to_q|to_k|to_v|to_out.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)\"",
        f"--gradient_accumulation_steps {gradient_accumulation_steps}",
        "--gradient_checkpointing",
        f"--checkpointing_steps {checkpoint_interval}",
        f"--checkpointing_limit {max_checkpoints}",
        "--enable_slicing",
        "--enable_tiling",
    ]

    if full:
        # remove --lora_alpha and --rank and --target_modules
        training_cmd = [cmd for cmd in training_cmd if cmd.count("--lora_alpha") == 0]
        training_cmd = [cmd for cmd in training_cmd if cmd.count("--rank") == 0]
        training_cmd = [cmd for cmd in training_cmd if cmd.count("--target_modules") == 0]
        training_cmd = [cmd for cmd in training_cmd if cmd.count("--training_type") == 0]
        training_cmd.append("--training_type full-finetune")

    if resume_checkpoint > 0:
        training_cmd.append(f"--resume_from_checkpoint {resume_checkpoint}")
    optimizer_cmd = [
        "--optimizer adamw",
        f"--lr {learning_rate}",
        "--lr_scheduler constant_with_warmup",
        "--lr_warmup_steps 200",
        "--lr_num_cycles 1",
        "--beta1 0.9",
        "--beta2 0.99",
        "--weight_decay 1e-4",
        "--epsilon 1e-8",
        "--max_grad_norm 1.0",
    ]
    validation_cmd = [
        f"--validation_dataset_file {default_validation_json}",
        f"--validation_steps {validation_interval}",
    ]
    other_cmd = [
        "--dataloader_num_workers 0",
        "--flow_weighting_scheme logit_normal",
        "--tracker_name finetrainers-flux",
        f"--output_dir {cache_dir}",
        "--init_timeout 600",
        "--nccl_timeout 600",
        "--report_to wandb",
    ]

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../3rdparty/finetrainers/train.py")
    print(src)

    if not online_mode:
        os.environ["WANDB_MODE"] = "offline"

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # run torch command
    cmd = [
        f"cd {directory} &&",
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "--rdzv_backend c10d",
        "--rdzv_endpoint=localhost:0",
        src,
        *parallel_cmd,
        *model_cmd,
        *dataset_cmd,
        *training_cmd,
        *optimizer_cmd,
        *validation_cmd,
        *other_cmd
    ]

    print(f"Args: {args}")
    print(f"Running command: {' '.join(cmd)}")
    with open("/tmp/tmp.txt", "w") as f:
        f.write(" ".join(cmd))
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
