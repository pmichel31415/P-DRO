#!/usr/bin/env python3
import os.path
import numpy as np
import torch as th
from src.configuration import Experiment, ArgumentGroup


def load_model_and_adversaries(folder, exp_name, suffix=None, use_accs=False):
    # Load the model losses
    suffix = "" if suffix is None else f"_{suffix}"
    losses_file = os.path.join(folder, f"{exp_name}_valid_results{suffix}.npz")
    losses_results = np.load(losses_file, allow_pickle=True)
    y = losses_results["y"]
    y_hat = losses_results["y_hat"]
    errors = 1-np.equal(y_hat, y).astype(float)
    log_p_hat = losses_results["log_p"]
    losses = np.stack([-log_p_hat[i, y[i]] for i in range(len(y))])
    # Load the weight corresponding to the adversaries
    advs_file = os.path.join(folder, f"{exp_name}.npz")
    advs = np.load(advs_file, allow_pickle=True)
    try:
        log_q = advs["dev_log_probs"]
    except KeyError:
        # Backward compatibility
        log_q = advs["dev_log_qs"]
    # Select nominal distribution
    if "dev_log_q0" in advs:
        log_q = np.concatenate([advs["dev_log_q0"], log_q])
    if len(log_q) == len(advs["all_dev_losses"]):
        log_q = np.concatenate([np.zeros((1, log_q.shape[1])), log_q])
    # Nominal distribution
    log_p = log_q[:1]
    # Compute weights log( q / p )
    log_weights = log_q - log_p
    # Robust errors
    domain_accs = np.stack(
        list(np.atleast_1d(losses_results["domain_scores"])[0].values()), -1)
    # Exp name
    exp_name = getattr(advs, "exp_name", "NAMELESS_EXP")
    return exp_name, losses, log_weights, log_p[0], domain_accs, errors


def test_metrics(save_name, select_by="adv", split="test"):
    results_file = os.path.join(f"{save_name}_valid_results_{select_by}.npz")
    results = np.load(results_file, allow_pickle=True)
    y = results["y"]
    y_hat = results["y_hat"]
    accuracy = (y == y_hat).astype(float).mean()
    domain_scores = np.atleast_1d(results["domain_scores"])[0]
    robust_accuracy = np.min(list(domain_scores.values()))
    return accuracy, robust_accuracy


def compare_models(
    losses: np.ndarray,
    weights: np.ndarray,
    baseline: float = 0,
):
    """Compare two models trained with parametric DRO

    This takes as input the losses of all models for each example of a dev set,
    as well as the weights of the examples according to their respective
    best adversary(ies)

    Args:
        losses: M x N array where losses[i, j] is the loss of
            model i on example j
        weights: K x N array where weights[i, j] is the weight of
            example j according to adversary i. The number of adversaries needs
            not match the number of models
    """
    N = losses.shape[1]
    # Compute the K by M loss matrix
    loss_matrix = weights.dot(losses.T) / N
    if baseline:
        loss_matrix -= weights.mean(-1, keepdims=True)
    # Choose best model by minmax
    best_model = np.argmin(loss_matrix.max(0))
    best_adv = np.argmax(loss_matrix[:, best_model])
    return best_model, best_adv


def filter_valid_advs(
    log_weights: np.ndarray,
    filter_advs_by: str,
    adv_threshold: float,
):
    # Ensure input shape is n_adversaries x n_examples
    if len(log_weights.shape) != 2:
        raise ValueError(
            "log_weights should have shape (n_adversaries, n_examples), got "
            f"{log_weights.shape} instead.")
    # Filter args
    if filter_advs_by == "none":
        # No filtering: return all true
        return np.ones(len(log_weights)).astype(bool)
    elif filter_advs_by == "reverse_kl":
        # Reverse KL
        adv_scores = (np.exp(log_weights) * log_weights).mean(-1)
    elif filter_advs_by == "alpha_coverage":
        # Alpha-coverage: max ratio
        adv_scores = np.exp(log_weights).max(-1)
    return adv_scores <= adv_threshold


def select_model(
    save_names,
    filter_advs_by="none",
    adv_threshold=np.inf,
    adv_threshold_num=None,
    baseline=0,
    verbose=False,
    select_by="adv",
    best_only=False,
    use_accs=False,
    group_idxs=None,
    renorm=False,
):
    # This will hold the model's losses and the adversaries' log_weights
    model_names = []
    advs_names = []
    losses = []
    errors = []
    domain_accs = []
    log_weights = []
    # Load models and adversaries
    for save_name in save_names:
        # Load model dev losses and adversaries log weights
        (
            exp_name,
            losses_model,
            log_weights_advs,
            log_p,
            domain_accs_model,
            errors_model,
        ) = load_model_and_adversaries(".", save_name, suffix=select_by)
        model_names.append(exp_name)

        if best_only:
            # Get the best model for each run
            valid_advs = filter_valid_advs(
                log_weights_advs, filter_advs_by, adv_threshold)
            log_weights_advs = log_weights_advs[valid_advs]
            _, best_adv = compare_models(losses_model.reshape(
                1, -1), log_weights_advs, baseline=0)
            log_weights_advs = log_weights_advs[best_adv].reshape(1, -1)
        # log_weights_advs = log_weights_advs[1:]
        # track adversaries for this experiment
        n_advs = len(log_weights_advs)
        # Record adversary name
        advs_names.extend([f"{exp_name}-{idx}" for idx in range(n_advs)])
        losses.append(losses_model)
        domain_accs.append(domain_accs_model)
        errors.append(errors_model)
        log_weights.append(log_weights_advs)
    # Convert to numpy arrays
    model_names = np.asarray(model_names)
    advs_names = np.asarray(advs_names)
    losses = np.stack(losses, axis=0)
    domain_accs = np.stack(domain_accs, axis=0)
    errors = np.stack(errors, axis=0)
    log_weights = np.concatenate(log_weights, axis=0)
    # Select best model
    if select_by == "adv":
        # Filter args
        if filter_advs_by != "none":
            # Filter out some adversaries
            if filter_advs_by == "reverse_kl":
                # This is equivalent to calling .mean() but maybe more stable
                # because we apply the renormalizer inside the exp()
                weights_ = np.exp(log_weights - np.log(log_weights.shape[-1]))
                adv_scores = (weights_ * log_weights).sum(-1)
            elif filter_advs_by == "alpha_coverage":
                # Filter out by max ratio
                adv_scores = np.exp(log_weights).max(-1)
            # If we have NaNs then the adversary is not valid
            adv_scores = np.nan_to_num(adv_scores, nan=adv_threshold+1)
            # Select valid adversaries
            valid_advs = adv_scores <= adv_threshold
            advs_names = advs_names[valid_advs]
            log_weights = log_weights[valid_advs]
            if verbose:
                print(valid_advs)
                print(adv_scores)
        # Compute adversarial weights
        weights = np.exp(log_weights)
        # Renormalize (before or after thresholding?)
        if renorm:
            weights = weights / weights.mean(-1, keepdims=True)
        # Now do the comparison
        if use_accs:
            # Compare using error/accuracy
            best_model_idx, p_advs = compare_models(errors, weights, baseline)
        else:
            # compare using losses
            best_model_idx, p_advs = compare_models(losses, weights, baseline)
    elif select_by == "erm":
        best_model_idx = np.argmin(errors.mean(-1))
    elif select_by == "robust":
        if group_idxs is None:
            # Use domain accuracies from the experiment run
            domain_errors = 1-domain_accs
        else:
            # Use custom domains
            assert sum(map(len, group_idxs.values())) == errors.shape[-1]
            domain_errors = np.stack(
                [errors[:, idxs].mean(-1) for idxs in group_idxs.values()],
                axis=-1,
            )
        # Minmax domain accuracy
        best_model_idx = np.argmin(domain_errors.max(-1))
    # Print results
    if verbose:
        print(f"The best model is {model_names[best_model_idx]}")
        print(f"\t(save file: {save_names[best_model_idx]})")
    return save_names[best_model_idx]


def get_args():
    experiment = Experiment("Adversarial DRO")
    # Experimental setting
    args = ArgumentGroup("General")
    args.add_argument("--random-seed", type=int, default=245778)
    args.add_argument("--n-reruns", type=int, default=None,
                      help="Number of reruns (with different seeds)")
    args.add_argument("--no-cuda", action="store_true",
                      help="Force CPU")
    args.add_argument("--save-names", type=str, nargs="+")
    args.add_argument("--train-log-interval", type=int, default=1)
    args.add_argument("--select-by", type=str, default="adv",
                      choices=["adv", "erm", "robust"])
    args.add_argument("--filter-advs-by", type=str, default="none",
                      choices=["none", "reverse_kl", "alpha_coverage"])
    args.add_argument("--adv-threshold", type=float, default=np.inf)
    args.add_argument("--adv-threshold-num", type=int, default=None)
    args.add_argument("--use-accs", action="store_true",
                      help="Use accuracies rather than loss")
    experiment.add_configuration(args)
    # Parse arguments
    experiment.parse_args()
    return experiment._configs_by_name


def main():
    configs = get_args()
    args = configs["General"]
    # Fix random seed
    np.random.seed(args.random_seed)
    th.random.manual_seed(args.random_seed+1)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # Run
    if args.n_reruns is None:
        best_save = select_model(
            args.save_names,
            args.filter_advs_by,
            args.adv_threshold,
            args.adv_threshold_num,
            select_by=args.select_by,
            use_accs=args.use_accs,
            verbose=False,
        )
        print(best_save)
        average, robust = test_metrics(
            best_save, select_by=args.select_by)
        print(average, robust)
    else:
        average = 0
        robust = 0
        for run_id in range(args.n_reruns):
            best_save = select_model(
                [f"{save_name}_run_{run_id}" for save_name in args.save_names],
                args.filter_advs_by,
                args.adv_threshold,
                args.adv_threshold_num,
                select_by=args.select_by,
                use_accs=args.use_accs,
                verbose=False,
            )
            print(best_save)
            run_average, run_robust = test_metrics(
                best_save, select_by=args.select_by)
            print(run_average, run_robust)
            average += run_average
            robust += run_robust
        print(average/args.n_reruns, robust/args.n_reruns)


if __name__ == "__main__":
    main()
