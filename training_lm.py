#!/usr/bin/env python3
"""
Trains a language model
"""
from src.models import build_model, ModelWithHead, architecture_list
from src.tasks import task_list, prepare_task
from argparse import ArgumentParser
import traceback
import numpy as np
import torch as th
import os.path
import tqdm
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from src.data import ByTokensSampler
from src.optim import optimizers, lr_schedulers
from src.optim import get_optimizer


def train(
    model,
    task,
    optimizer,
    n_steps=100,
    n_epochs=None,
    batch_size=64,
    max_tokens_per_batch=None,
    update_every=1,
    clip_grad=-1,
    lr=1e-3,
    mom=0,
    weight_decay=1e-5,
    valid_interval=50,
    train_log_interval=1,
    device="cuda:0",
    exp_name="",
    figure_prefix="precisions",
    results_prefix="results/",
    lower_is_better=False,
    lr_scheduler="constant",
):
    # Save file
    model_file = os.path.join(results_prefix, f"{exp_name}_model.pt")
    # Train data
    # Batch sampler
    if max_tokens_per_batch is not None:
        # If we are dealing with text
        sampler = ByTokensSampler(
            task.train_data,
            max_samples=batch_size,
            max_tokens=max_tokens_per_batch,
            shuffle=True,
        )
    else:
        sampler = BatchSampler(
            RandomSampler(task.train_data),
            batch_size=batch_size,
            drop_last=False,
        )
    # Dataloader
    loader = DataLoader(
        task.train_data,
        batch_sampler=sampler,
        collate_fn=task.collate_fn,
    )
    # Number of steps and epochs
    steps_per_epochs = len(sampler)
    if n_epochs is not None:
        n_steps = None  # Don't stop based on step
    else:
        # Make sure we run as many epochs as necessary to reach
        # the appropriate number of steps
        n_epochs = int(np.ceil(n_steps/steps_per_epochs))
    # Validate by epoch maybe?
    if valid_interval == "epoch":
        valid_interval = None
    else:
        valid_interval = int(valid_interval)
    # Step tracker
    step = 0
    # set the model's mode to training mode.
    model.train()
    # Optimizer for this task
    opt = get_optimizer(
        optimizer,
        list(model.parameters()),
        lr=lr,
        mom=mom,
        weight_decay=weight_decay,
    )
    # LR schedule
    # scheduler = get_lr_scheduler(lr_scheduler, opt, step, n_steps)
    # Track current overall step, scores, etc
    eval_steps = []
    dev_scores = [np.inf if lower_is_better else 0]
    # Training loop
    for epoch in range(1, n_epochs+1):
        # Data source
        itr = tqdm.tqdm(loader)
        for step_in_epoch, batch in enumerate(itr, 1):
            # Total step
            step += 1
            # Reset gradients
            if (step-1) % update_every == 0:
                opt.zero_grad()
            # Get data
            batch = batch.to(device)
            # run the model and backpropagate the errors.
            nll = task.nll(model, batch)
            # Backward pass
            nll.backward()
            # Take a step
            if step % update_every == 0:
                # Clip model gradient
                if clip_grad > 0:
                    th.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                opt.step()
                # scheduler.step()
            # Periodically check valid accuracy
            if (
                # Either we validate every fixed number of step
                (valid_interval is not None and step % valid_interval == 0)
                # Or we validate every epoch
                # (so check that we reached the end of this epoch)
                or (valid_interval is None and step_in_epoch == len(sampler))
            ):
                # count this iteration
                eval_steps.append(step)
                # Run on the dev set
                current_dev_score = task.eval_model(model, data="valid")
                # Test source and target task
                dev_scores.append(current_dev_score)
                # Dump to npz
                np.savez_compressed(
                    f"{results_prefix}{exp_name}.npz",
                    eval_steps=eval_steps,
                    dev_scores=dev_scores,
                    name=task.name,
                )
                # Is the model the best yet?
                if lower_is_better:
                    best_model_yet = min(dev_scores[:-1]) > dev_scores[-1]
                else:
                    best_model_yet = max(dev_scores[:-1]) < dev_scores[-1]
                # If so, save it now
                if len(dev_scores) == 1 or best_model_yet:
                    print(f"Saving best model to {model_file}")
                    th.save(model.state_dict(), model_file)
            # Update log
            if step % train_log_interval == 0:
                # Track GPU infos
                gpu_str = ""
                if device is not None:
                    # Reserved mempry (in GB)
                    reserved = th.cuda.memory_reserved(device) / 1e9
                    gpu_str = f" GPU mem: {reserved:.1f}GB"
                # Update iterator description
                itr.set_description((
                    f"Epoch {epoch} => "
                    f"loss: {nll.item():.3f} "
                    f"dev score (best): {dev_scores[-1]:.1f}"
                    f" ({min(dev_scores):.1f})"
                    f"{gpu_str}"
                ))
            # Stop based on step
            if n_steps is not None and step >= n_steps:
                break
    # Test best model on both tasks
    best_model_state_dict = th.load(model_file)
    model.load_state_dict(best_model_state_dict)
    # Test
    return model


def get_args():
    parser = ArgumentParser("Training and evaluating a language model")
    # Experimental setting
    parser.add_argument("--random-seed", type=int, default=245778)
    parser.add_argument("--n-reruns", type=int, default=1,
                        help="Number of reruns (with different seeds)")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU")
    parser.add_argument("--test-only", action="store_true", help="Just test")
    parser.add_argument("--train-log-interval", type=int, default=1)
    parser.add_argument("--valid-interval", default=250)
    parser.add_argument("--exp-prefix", type=str, default="")
    parser.add_argument("--task", default="MNIST",
                        choices=list(task_list.keys()),)
    parser.add_argument("--grid-search-configs", type=str, nargs="*",
                        help="string describing the range over which to "
                        "perform hyper-parameter search. Format: "
                        "param_1=val_1,val_2[...] param_2=val_1[...]",
                        default=[])
    # Model specific arguments
    parser.add_argument('--architecture', type=str, default="ff_lm",
                        choices=list(architecture_list.keys()))
    parser.add_argument('--input-format', type=str, default=None,
                        choices=[None, "bert-base-uncased", "gpt2"],
                        help="Format (tok+vocabulary) for text input. "
                        "If None: decided based on the architecture.")
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--hidden-dropout-prob", type=float, default=.5)
    parser.add_argument("--input-dropout-prob", type=float, default=.2)
    parser.add_argument("--load-model-from-file", default=None, type=str)
    # Optimization arguments
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=list(optimizers.keys()))
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--mom", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens-per-batch", type=int, default=2000)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--clip-grad", type=float, default=10.0)
    parser.add_argument("--lower-is-better", action="store_true",
                        help="By default the model with highest dev score will"
                        " be saved: this is to specifiy that the model with "
                        "the lowest score should be saved instead. This is for"
                        " tasks like language modeling where perplexity is "
                        "used.")
    parser.add_argument("--lr-scheduler", type=str, default="constant",
                        choices=list(lr_schedulers.keys()))
    # Task specific arguments
    return parser.parse_args()


def make_exp_name(args, run_id=None):
    """Make a name for this experiment given the arguments"""
    exp_name = []
    if getattr(args, "exp_prefix", "") != "":
        exp_name.append(args.exp_prefix)
    exp_name.append(args.task)
    exp_name.append(args.architecture)
    if run_id is not None:
        exp_name.append(f"run_{run_id}")
    return "_".join(exp_name)


def main():
    args = get_args()
    # decide whether to use cuda or not.
    if th.cuda.is_available() and not args.no_cuda:
        device = th.device("cuda:0")
    else:
        device = th.device("cpu")
    # Fix random seed
    np.random.seed(args.random_seed)
    th.random.manual_seed(args.random_seed+1)

    # Retrieve input format
    # This is import for textual data because we need to know how to
    # preprocess the data (eg. tokenize, split into subwords)
    input_format = args.input_format
    if input_format is None:
        input_format = args.architecture
    # Retrieve the task
    task, input_shape, output_size = prepare_task(
        args.task,
        model_name=input_format,
    )
    print(task.__class__.__name__)
    # Now evaluate on the test tasks for a number of runs
    for run_id in range(args.n_reruns):
        # Fix random seed
        np.random.seed(args.random_seed+run_id)
        th.random.manual_seed(args.random_seed+run_id+1)
        # Experiment name
        exp_name = make_exp_name(args, run_id if args.n_reruns > 1 else None)
        # Prepare the model
        model = build_model(args.architecture, input_shape, output_size)
        head = task.create_compatible_head(model.hidden_size)
        model = ModelWithHead(model, head)
        # Load model weights as needed
        if args.load_model_from_file is not None:
            saved_model_state_dict = th.load(args.load_model_from_file)
            model.load_state_dict(saved_model_state_dict)
        # Cast model to device
        model = model.to(device)
        # Print all arguments
        for k, v in sorted(args.__dict__.items(), key=lambda x: x[0]):
            print(k, ":", v)
        print("="*100, flush=True)
        # Training
        if not args.test_only:
            model = train(
                model,
                task,
                optimizer=args.optimizer,
                lr_scheduler=args.lr_scheduler,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                max_tokens_per_batch=args.max_tokens_per_batch,
                update_every=args.update_every,
                clip_grad=args.clip_grad,
                lr=args.lr,
                mom=args.mom,
                weight_decay=args.weight_decay,
                train_log_interval=args.train_log_interval,
                valid_interval=args.valid_interval,
                device=device,
                exp_name=exp_name,
                figure_prefix="figures",
                results_prefix="results/",
                lower_is_better=args.lower_is_better,
            )
        # Test
        print(f"Testing on task {task.name}", flush=True)
        score = task.eval_model(model, data="test")
        print(f"Score: {score:.4f}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit()
