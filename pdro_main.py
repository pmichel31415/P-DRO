#!/usr/bin/env python3
"""
Main program
"""
from src.models import build_model, ModelWithHead
from src.tasks import task_list, prepare_task
import traceback
import numpy as np
import torch as th
import torch.nn.functional as F
import os.path
import tqdm
import hashlib
import scipy.optimize
from typing import Optional, Tuple
from src.data.language_modeling import to_lm_batch
from src.optim import get_optimizer, get_lr_scheduler
from src.utils import cacheable, get_loader, get_group_dro_loader
from src.tasks import LanguageModelingTask, CCLanguageModelingTask, Task
from src.configuration import Experiment, ArgumentGroup
from src.running_average import get_log_running_average, LogRunningAverage
from src.stopping import (
    AverageStopping,
    GreedyMinMaxStopping,
    GroupRobustStopping,
)
from src.logging import NpzLogger

import non_param_dro
import pdro_args
from pdro_compare_models import filter_valid_advs


@cacheable(format="pt")
def compute_dataset_log_probs(
    lm,
    task,
    dataset="train",
    batch_size=64,
    max_tokens_per_batch=None,
    joint=False,
    class_conditional=False,
    ratio_model=False,
    num_workers=1,
):
    """Compute log probability of every sample in a dataset

    Args:
        lm (nn.Module): language model
        task (Task): language modeling task (for computing the loss function)
        dataset (str, optional): Dataset. Defaults to "train"
            (training data of the task).
        batch_size (int, optional): Batch size. Defaults to 64.
        max_tokens_per_batch (int, optional): Number of tokens per batch
            (for text data). Defaults to None.

    Returns:
        torch.Tensor: Tensor containing all scores
    """
    # LM task
    if ratio_model:
        # If using a ratio model we don't need to modify the task
        adv_task = task
    elif joint or class_conditional:
        # if using a joint/generative (q(x, y)) or class-conditional (q(x|y))
        # adversary, transform to class-conditional LM task
        adv_task = CCLanguageModelingTask.from_text_task(
            task, generative=not class_conditional)
    else:
        # Otherwise transform to LM task
        adv_task = LanguageModelingTask.from_text_task(task)
    # Snapshot mode and set to eval mode
    mode = lm.training
    lm.train(mode=False)
    # Determine dataset
    if dataset == "train":
        dataset = adv_task.train_data
    elif dataset == "valid":
        dataset = adv_task.valid_data
    elif dataset == "test":
        dataset = adv_task.test_data
    elif not isinstance(dataset, th.utils.data.Dataset):
        raise ValueError(
            "dataset should be either a pytorch Dataset or one of"
            "'train', 'valid', 'test'"
        )
    # Dataloader
    sampler, loader = get_loader(
        dataset,
        batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=False,
        collate_fn=adv_task.collate_fn,
        num_workers=num_workers,
    )
    # Computing all nlls
    all_nlls = []
    for batch in tqdm.tqdm(loader, desc="Computing LM scores"):
        with th.no_grad():
            if ratio_model:
                logits = adv_task.logits(lm, batch)
                y = batch.outputs.to(logits.device)
                nlls = F.nll_loss(logits, y, reduction="none")
            else:
                nlls = adv_task.nll(lm, batch, reduction="none").sum(-1)
            all_nlls.append(nlls.clone().detach().cpu())
    all_nlls = th.cat(all_nlls)
    lm.train(mode=mode)
    return -all_nlls.clone().detach()


def find_tau_star(ell, kappa, log_min=-10, log_max=10):

    # Find \tau^* such that KL(q^*_\tau^* || p) = \kappa
    # Heuristically we've found that values of \tau can be very small (<10^2)
    # or sometimes big (10^2). Therefore, searching for \log_10 \tau^* is
    # marginally faster since the values taken by \tau^* are more evenly
    # spread out on the log scale

    def kl_diff(log_tau):
        tau = 10**log_tau
        log_q_star_ = ell/tau
        log_q_star_ -= log_q_star_.max()
        log_q_star = log_q_star_ - np.log(np.mean(np.exp(log_q_star_)))
        return (np.exp(log_q_star)*log_q_star).mean() - kappa
    # Check boundary conditions
    if kl_diff(log_min)*kl_diff(log_max) >= 0:
        # In this case \tau^* lies outside the interval
        if np.abs(kl_diff(log_min)) < np.abs(kl_diff(log_max)):
            # \tau^* lies to the left of the interval so the minimum value
            # is our best guess
            log_tau_star = log_min
        else:
            # Or it lies to the right
            log_tau_star = log_max
    else:
        # \tau^* lies inside the interval, use the bisection method
        # (=binary search) to find it
        log_tau_star = scipy.optimize.bisect(
            kl_diff, log_min, log_max, xtol=1e-5, maxiter=100)

    return 10**(log_tau_star)


def compute_model_loss(
    losses: th.Tensor,
    log_q: th.Tensor,
    log_p: th.Tensor,
    adv_args: ArgumentGroup,
    log_Z_adv: LogRunningAverage,
    log_Z_model: LogRunningAverage,
    errors: Optional[th.Tensor],
) -> th.Tensor:
    """Computes the loss of the model given the model's los and the
    adversary's weights on each sample

    Args:
        losses: Loss of each sample (of shape [B])
        log_q: Log probability of each sample under the adversary
            (of shape [B])
        log_p: Log probability of each sample under the MLE model
            (of shape [B])
        adv_args: Arguments for the adversary
        log_Z_adv: Log normalizer for the adversary's weights
        log_Z_model: Log normalizer for the model's weights
        errors: some arbitrary error function of the model's output on each
        sample (of shape [B])

    Returns:
        Loss for the model on this batch (a scalar tensor)
    """
    # Compute the log ratio
    if adv_args.non_param:
        # Non-parametric adversaries
        if adv_args.chi2_eta is not None:
            q_star = non_param_dro.Chi2ConstrainedAdversary(
                adv_args.chi2_eta).best_response(losses)
        elif adv_args.cvar_alpha is not None:
            q_star = non_param_dro.CVaRConstrainedAdversary(
                adv_args.cvar_alpha).best_response(losses)
        elif adv_args.kappa is not None:
            # If the KL bound is fixed, we find the temperature which
            # satisfies it
            q_star = non_param_dro.KLConstrainedAdversary(
                adv_args.kappa).best_response(losses)
        else:
            # Otherwise just use a fixed temperature
            tau_star = adv_args.tau
            # Un-normalized q_star
            log_q_star_ = losses/tau_star
            # log normalize
            # Note that the log normalizer is
            # E_z~p e^{l(z)/\tau} which we approximate with the empirical
            # average of e^{l/\tau} over the minibatch
            log_Z = th.logsumexp(log_q_star_ - np.log(len(losses)), dim=0)
            log_q_star = log_q_star_ - log_Z
            # Weights for the loss function
            # Notice that we don't detach tthe weights: we will backprop
            # through q_\tau,\theta too
            q_star = th.exp(log_q_star)
        # Compute the model loss
        model_loss = (q_star*losses).sum()
    else:
        log_ratios = log_q-log_p
        # Renormalize weights
        log_ratios = log_ratios - log_Z_model.value
        # Importance weights
        weights = th.exp(log_ratios)
        # Loss
        model_loss = (weights.detach()*losses).sum()
    # Interpolate between the adversarial loss and the ERM objective
    # 1 means we are only training on the adversarial objective
    # 0 means we are only training on the ERM objective
    if adv_args.alpha < 1:
        erm_loss = losses.mean()
        model_loss = model_loss*adv_args.alpha + (1-adv_args.alpha)*erm_loss

    return model_loss


def compute_adv_loss(
    losses: th.Tensor,
    log_q: th.Tensor,
    log_p: th.Tensor,
    adv_args: ArgumentGroup,
    log_Z_adv: LogRunningAverage,
    log_Z_model: LogRunningAverage,
    errors: Optional[th.Tensor],
) -> th.Tensor:
    """Compute the adversary's loss given the model's loss on a batch of
    examples and the weights produced by the adversary

    Args:
        losses: A tensor containing the losses of the model on a
            minibatch
        log_q: A tensor containing the probability of each example
            in the mininbatch
        log_p: A tensor containing the baseline probability for
            each example in the batch
        adv_args: Arguments specific to the adversary
        log_Z_adv: Running average of the weights used in
            computing the adversary's loss
        errors: Tensor containing the errors of the model on the
            minibatch (these can be non-differentiable, as opposed as the
            losses)
        log_Z_model: This is the log normalizer of the
            weights used to compute the model's loss. Here this is used to
            recompute the model loss in the `zero_sum` setting (where the
            adversary is trained to maximize the model's loss)
    """
    # Interpolate with the regular nll
    if adv_args.non_param:
        # Non parametric case: we don't train the adversary
        return th.zeros(1, requires_grad=True)
    elif adv_args.adv_obj == "zero_sum":
        # LM NLL in log space:
        weights = (log_q - log_p) - log_Z_model.value
        adv_loss = -(th.exp(weights)*losses.detach()).mean()
    elif adv_args.adv_obj == "fwd_kl":
        # Log likelihood ratio
        log_weights = (log_q - log_p) - log_Z_model.value
        # weights
        weights = th.exp(log_weights)
        # "KL penalty" component
        kl_loss = (weights*log_weights).mean()
        # "zero sum" component
        zero_sum_loss = (-weights*losses.detach()).mean()
        adv_loss = zero_sum_loss + adv_args.tau*kl_loss
    elif adv_args.adv_obj == "log_zero_sum":
        # LM NLL in log space:
        log_losses = log_q - log_p + th.log(losses).detach()
        adv_loss = -th.logsumexp(log_losses, 0)
    elif adv_args.adv_obj.startswith("exp"):
        if adv_args.adv_on_acc:
            log_q_star = errors / adv_args.tau
        else:
            # q*(x, y) \propto \ell(x, y)/temp * p
            log_q_star = losses.detach() / adv_args.tau
        if adv_args.adv_obj == "exp":
            # Reweight by log_p
            log_lm_weights = log_q_star-log_p
        elif adv_args.adv_obj == "exp_kl":
            # Reweight by log_p
            log_lm_weights = log_q_star
        # Actual weights are normalized across minibatch
        log_normalizer = th.logsumexp(log_lm_weights, 0).item()
        # Running average
        log_Z_adv += log_normalizer
        # print(log_Z_adv.value, flush=True)
        # log_lm_weights += np.log(batch.size)
        lm_weights = th.exp(log_lm_weights-log_Z_adv.value)
        # Loss for the lm
        adv_loss = -(lm_weights*log_q).sum()
    # # lm_loss = -(th.exp(log_q-log_p)*nlls.detach()).mean()
    if adv_args.ratio_model and adv_args.self_norm_lambda > 0:
        log_expected_ratio = th.logsumexp(log_q-np.log(len(log_q)), dim=0)
        adv_loss = adv_loss + adv_args.self_norm_lambda*log_expected_ratio**2
    # Interpolate with the likelihood of the data
    # (this pulls back the adversary towards the nominal distribution)
    if adv_args.beta < 1:
        adv_loss = adv_args.beta*adv_loss + (1-adv_args.beta) * (-log_q).mean()
    return adv_loss


def train(
    model: th.nn.Module,
    adv: th.nn.Module,
    task: Task,
    model_args: ArgumentGroup,
    adv_args: ArgumentGroup,
    optim_args: ArgumentGroup,
    dro_args: ArgumentGroup,
    train_log_interval: int = 1,
    device="cuda:0",
    exp_name: str = "",
    figure_prefix: str = "precisions",
    results_prefix: str = "results/",
    eval_domain_filters=None,
    train_domain_filters=None,
    valid_pseudo_domain_filters=None,
    save_name: str = "",
):
    # LM task
    if adv_args.ratio_model:
        adv_task = task
    elif adv_args.joint or adv_args.class_conditional:
        adv_task = CCLanguageModelingTask.from_text_task(
            task, generative=not adv_args.class_conditional)
    else:
        adv_task = LanguageModelingTask.from_text_task(task)
    # Save files
    model_file = os.path.join(results_prefix, f"{save_name}_model.pt")
    lm_file = os.path.join(results_prefix, f"{save_name}_lm.pt")
    adv_model_file = os.path.join(results_prefix, f"{save_name}_adv_model.pt")
    adv_lm_file = os.path.join(results_prefix, f"{save_name}_adv_lm.pt")
    robust_model_file = os.path.join(
        results_prefix, f"{save_name}_robust_model.pt")
    robust_lm_file = os.path.join(results_prefix, f"{save_name}_robust_lm.pt")
    # Optimizer for this task
    opt = get_optimizer(
        optim_args.optimizer,
        list(model.parameters()),
        lr=optim_args.lr,
        weight_decay=optim_args.weight_decay,
    )
    # Optimizer for the adversary
    # Default to the model's optimizer
    adv_optimizer = adv_args.adv_optimizer
    if adv_optimizer is None:
        adv_optimizer = optim_args.optimizer
    adv_opt = get_optimizer(
        adv_optimizer,
        list(adv.parameters()),
        lr=optim_args.lr if adv_args.adv_lr is None else adv_args.adv_lr,
        mom=adv_args.adv_mom,
        weight_decay=optim_args.weight_decay,
    )
    # Gradient clipping for adversary defaults to model's clip rate
    if adv_args.clip_grad_adv is None:
        adv_args.clip_grad_adv = optim_args.clip_grad

    # Log normalizers
    log_Z_model = get_log_running_average(adv_args.norm_k_model)
    log_Z_adv = get_log_running_average(adv_args.norm_k_adv)
    # Indices for each pseudo domain on the training set (if available)
    q = np.zeros(1)
    # Setup group DRO
    if train_domain_filters is not None:
        # Get the subset of the training data corresponding to each domain
        train_domains_data = {
            domain: (
                task.train_data.filtered(domain_filter)
                if callable(domain_filter) else
                task.train_data.subset(domain_filter)
            )
            for domain, domain_filter in train_domain_filters.items()
        }
        # Get the name of the domains
        train_domains = list(sorted(train_domain_filters.keys()))
        # Set up Group DRO
        # Initialize the weight q of each domain to the uniform distribution
        q = np.ones(len(train_domains))/len(train_domains)
        # Baseline scores for each domain
        baseline = dro_args.baseline
        if baseline is None:
            baseline = np.zeros(len(train_domains))
        # Add domain information to the training samples directly
        for group_idx, domain in enumerate(train_domains):
            domain_data = train_domains_data[domain]
            # FIXME: this could be removed
            if domain_data.attributes == []:
                domain_data.attributes = [{} for _ in domain_data]
            for idx in range(len(domain_data)):
                domain_data.attributes[idx]["group_idx"] = group_idx
    # Validation pseudo domains (this is for Topic CVaR)
    if valid_pseudo_domain_filters is not None:
        valid_pseudo_domains_idxs = {
            domain: ([idx for idx, x in enumerate(task.valid_data.attributes)
                      if domain_filter(x)]
                     if callable(domain_filter) else
                     domain_filter)
            for domain, domain_filter in valid_pseudo_domain_filters.items()
        }
    # Indices for each domain in the dev set (to evalu)
    if eval_domain_filters is not None:
        # Domain index of each sample
        domain_idxs = {
            domain: [idx for idx, x in enumerate(task.valid_data.attributes)
                     if domain_filter(x)]
            for domain, domain_filter in eval_domain_filters.items()
        }

    # Stopping
    avg_stop = AverageStopping(lower_is_better=False)
    # Minmax stopping
    adv_stop = GreedyMinMaxStopping(len(task.valid_data), lower_is_better=True)
    # Add ERM weights
    adv_stop.add_adv_log_weights(np.zeros(len(task.valid_data)))
    # Min max on pseudo groups instead
    if valid_pseudo_domain_filters is not None:
        adv_stop = GroupRobustStopping(
            valid_pseudo_domains_idxs,
            lower_is_better=False
        )
    # Group robust stopping
    robust_stop = GroupRobustStopping(domain_idxs, lower_is_better=False)
    # Compute the initial log probabilities of the adversary on the dev set
    if not adv_args.ratio_model:
        dev_log_q0 = compute_dataset_log_probs(
            adv,
            task,
            "valid",
            optim_args.batch_size,
            optim_args.max_tokens_per_batch,
            joint=adv_args.joint,
            class_conditional=adv_args.class_conditional,
            ratio_model=adv_args.ratio_model,
            num_workers=optim_args.num_workers,
        ).numpy()
    else:
        dev_log_q0 = np.zeros(len(task.valid_data))
    dev_log_q = dev_log_q0.copy()
    # set the model's mode to training mode.
    model.train()
    # No dropout for the adversary (otherwise the likelihood ratios can become
    # bad). In an ideal scenario we would dropout the same weights both for
    # adversary and MLE model, but since the MLE log probabilities are
    # pre-computed with the full model we are out of luck
    adv.eval()
    # Train data
    # Dataloader
    sampler, loader = get_loader(
        task.train_data,
        optim_args.batch_size,
        max_tokens_per_batch=optim_args.max_tokens_per_batch,
        shuffle=True,
        collate_fn=task.collate_fn,
        num_workers=optim_args.num_workers,
    )
    # If training with DRO, we sample each group equiprobably
    if train_domain_filters is not None:
        sampler, loader = get_group_dro_loader(
            [train_domains_data[domain] for domain in train_domains],
            optim_args.batch_size,
            max_tokens_per_batch=optim_args.max_tokens_per_batch,
            shuffle=True,
            collate_fn=task.collate_fn,
            num_workers=optim_args.num_workers,
        )
    # Number of steps and epochs
    steps_per_epochs = len(sampler)
    if optim_args.n_epochs is not None:
        optim_args.n_steps = steps_per_epochs*optim_args.n_epochs
        # Don't stop based on step
        stop_by_step = False
    else:
        # Make sure we run as many epochs as necessary to reach
        # the appropriate number of steps
        stop_by_step = True
        optim_args.n_epochs = int(np.ceil(optim_args.n_steps/steps_per_epochs))
    # Validate by epoch maybe?
    if optim_args.valid_interval == "epoch":
        valid_interval = None
    else:
        valid_interval = int(optim_args.valid_interval)
    # Get lr scheduler
    lr_schedule = get_lr_scheduler(
        optim_args.lr_scheduler,
        opt,
        optim_args.lr,
        optim_args.n_steps,
    )
    # Logging
    log_tracker = NpzLogger(
        filename=f"{results_prefix}{save_name}.npz",
        static_fields={"exp_name": exp_name,
                       "name": task.name,
                       "dev_log_q0": dev_log_q0},
        overwrite=True,
    )
    # Step tracker
    step = 0
    # Training loop
    for epoch in range(1, optim_args.n_epochs+1):
        # Data source
        itr = tqdm.tqdm(loader)
        for step_in_epoch, batch in enumerate(itr, 1):
            # Total step
            step += 1
            # Reset gradients
            if (step-1) % optim_args.update_every == 0:
                opt.zero_grad()
            # if (step-1) % lm_update_every == 0:
            adv_opt.zero_grad()
            # Get data on device
            batch = batch.to(device)
            # Model forward pass to get the losses and predictions
            nlls, _, y_hat = task.nll(
                model,
                batch,
                reduction="none",
                predict=True,
            )
            # Model errors
            errors = (batch.outputs != y_hat).float().detach()
            # Adversary forward pass
            if adv_args.pdro:
                # Transform the minibatch for processing by the adversary
                lm_batch = batch
                if not (adv_args.joint or adv_args.class_conditional):
                    lm_batch = to_lm_batch(lm_batch)
                # Get log prob of each sample under the adversary
                if adv_args.ratio_model:
                    logits = adv_task.logits(adv, batch)
                    y = batch.outputs.to(logits.device)
                    log_q = - F.nll_loss(logits, y, reduction="none")
                    if adv_args.renorm_ratios:
                        log_q = th.log_softmax(
                            log_q, dim=0) + np.log(len(log_q))
                else:
                    # Get NLL for words
                    log_q = -adv_task.nll(adv, lm_batch, reduction="none")
                    # Sum along the length dimention
                    log_q = log_q.sum(-1)
                # log prob under the MLE LM
                log_p = th.tensor(batch.attributes["log_p"]).to(log_q.device)
                # Keep track of the log normalizer for the weights used to
                # compute the model's loss
                log_Z_model += th.logsumexp(log_q-log_p, 0).item()
                model_loss = compute_model_loss(nlls, log_q, log_p, adv_args,
                                                log_Z_adv, log_Z_model, errors)
                # Compute the adversary's loss
                adv_loss = compute_adv_loss(nlls, log_q, log_p, adv_args,
                                            log_Z_adv, log_Z_model, errors)
                # Backward pass
                adv_loss.backward()
            elif dro_args.group_dro:
                # Retrieve group indices for each sample
                group_idxs = th.tensor(batch.attributes["group_idx"]).long()
                # loss = average over q
                group_weights = th.tensor(q[group_idxs.numpy()])
                model_loss = (group_weights.to(nlls.device)*nlls).mean()
                # Group losses
                losses = np.nan_to_num(
                    [nlls[group_idxs.to(nlls.device) == idx].mean().item()
                     for idx in range(len(q))],
                    # NaN means there were no representatives of this group in
                    # the batch, so we return 0 instead
                    nan=0,
                )
                # Update q in log space
                log_q = np.log(q) + dro_args.eta_q*losses
                # Renormalize (this is basically a softmax)
                q -= log_q.max()
                q_ = np.exp(log_q)
                q = q_ / q_.sum()

            else:
                # ERM
                model_loss = nlls.mean()
            # L2 regularization for the model
            if optim_args.l2_reg > 0:
                param_vec = th.cat([p.view(-1) for p in model.parameters()])
                model_loss += optim_args.l2_reg * th.sum(param_vec**2)
            # Model backward pass
            model_loss.backward()
            # Take a step
            if step % optim_args.update_every == 0:
                # Clip model gradient
                if optim_args.clip_grad > 0:
                    th.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        optim_args.clip_grad,
                    )
                # Update params and LR
                opt.step()
                lr_schedule.step()
            if step % adv_args.adv_update_every == 0:
                # Clip adv gradient
                if adv_args.clip_grad_adv > 0:
                    th.nn.utils.clip_grad_norm_(
                        adv.parameters(),
                        adv_args.clip_grad_adv,
                    )
                # Update adversary
                adv_opt.step()
            # Periodically check valid accuracy
            if (
                # Either we validate every fixed number of step
                (valid_interval is not None and step % valid_interval == 0)
                # Or we validate every epoch
                # (so check that we reached the end of this epoch)
                or (valid_interval is None and step_in_epoch == len(sampler))
            ):
                # Compute the score for each individual dev example
                (dev_examples_scores, dev_losses) = task.eval_model(
                    model,
                    data="valid",
                    by_example=True,
                    nll=True,
                )
                dev_examples_scores = dev_examples_scores.numpy()
                dev_losses = dev_losses.numpy()
                # Save model if it beats the previous best model
                if avg_stop.update_best(dev_examples_scores):
                    th.save(model.state_dict(), model_file)
                    th.save(adv.state_dict(), lm_file)
                # Robust stopping
                if robust_stop.update_best(dev_examples_scores):
                    th.save(model.state_dict(), robust_model_file)
                    th.save(adv.state_dict(), robust_lm_file)
                # Adversarial stopping
                if adv_args.pdro:
                    valid_adv = adv
                    # For non-parametric adversary, we compute the adversarial
                    # weights directly with the loss
                    if adv_args.non_param:

                        # Non-parametric adversaries
                        if adv_args.chi2_eta is not None:
                            valid_adv = non_param_dro.Chi2ConstrainedAdversary(
                                adv_args.chi2_eta)
                            dev_q = valid_adv.best_response(
                                th.Tensor(dev_losses)).numpy()
                        elif adv_args.cvar_alpha is not None:
                            valid_adv = non_param_dro.CVaRConstrainedAdversary(
                                adv_args.cvar_alpha)
                            dev_q = valid_adv.best_response(
                                th.Tensor(dev_losses)).numpy()
                        elif adv_args.kappa is not None:
                            # If the KL bound is fixed, we find the
                            # temperature which satisfies it
                            valid_adv = non_param_dro.KLConstrainedAdversary(
                                adv_args.kappa)
                            dev_q = valid_adv.best_response(
                                th.Tensor(dev_losses)).numpy()
                        else:
                            # Or use
                            tau_star = adv_args.tau
                            dev_log_q = dev_losses/tau_star
                            # For stability for log sum exp
                            dev_log_q -= dev_log_q.max()
                            # Log normalize
                            dev_log_q -= np.log(np.sum(np.exp(dev_log_q)))
                            dev_q = np.exp(dev_log_q)
                        dev_log_q = np.log(dev_q+1e-40) + \
                            np.log(len(dev_log_q))
                    else:
                        # Compute weights with adversary
                        dev_log_q = compute_dataset_log_probs(
                            valid_adv,
                            task,
                            "valid",
                            optim_args.batch_size,
                            optim_args.max_tokens_per_batch,
                            joint=adv_args.joint,
                            class_conditional=adv_args.class_conditional,
                            ratio_model=adv_args.ratio_model,
                            num_workers=optim_args.num_workers,
                        ).numpy()
                    # First get the log importance weights of current adversary
                    log_weights = dev_log_q
                    # in regular p-dro we need to divide by the MLE likelihood
                    if not (adv_args.ratio_model or adv_args.non_param):
                        log_weights = log_weights - dev_log_q0
                    elif adv_args.ratio_model and adv_args.renorm_ratios:
                        log_weights -= log_weights.max()
                        log_weights -= np.log(np.sum(np.exp(log_weights)))
                        log_weights += np.log(len(log_weights))
                    # Filter out bad adversaries
                    is_valid = filter_valid_advs(
                        log_weights.reshape(1, -1),
                        adv_args.filter_advs_by,
                        adv_args.adv_threshold,
                    )
                    # If the adversary is valid, add it to the list for
                    # computing the inner max
                    if is_valid:
                        adv_stop.add_adv_log_weights(log_weights)
                    # These are the losses we'll use for computing minmax
                    # validation loss
                    adv_valid_scores = dev_losses
                    if adv_args.adv_valid_on_acc:
                        adv_valid_scores = 1-dev_examples_scores
                elif valid_pseudo_domain_filters is not None:
                    # FIXME
                    adv_valid_scores = dev_examples_scores
                else:
                    adv_valid_scores = dev_examples_scores

                # Save model if it beats the previous best model
                if adv_stop.update_best(adv_valid_scores):
                    th.save(model.state_dict(), adv_model_file)
                    th.save(adv.state_dict(), adv_lm_file)
                # Dump to npz
                log_tracker.append(
                    eval_steps=step,
                    dev_scores=avg_stop.current_score,
                    dev_log_qs=dev_log_q,
                    all_dev_examples_scores=dev_examples_scores,
                    all_dev_losses=dev_losses,
                    domain_scores=robust_stop._current_group_scores,
                    q=q,
                )
            # Update log
            if step % train_log_interval == 0:
                # Track GPU infos
                gpu_str = ""
                if device is not None and th.cuda.device_count() > 0:
                    # Reserved mempry (in GB)
                    reserved = th.cuda.memory_reserved(device) / 1e9
                    gpu_str = f" GPU mem: {reserved:.1f}GB"
                # DRO stats
                dro_str = ""
                if eval_domain_filters is not None:
                    dro_str = (
                        f" robust (best): {100*robust_stop.current_score:.1f}"
                        f" ({100*robust_stop.previous_best:.1f})"
                    )
                # Update iterator description
                itr.set_description((
                    "=> "
                    f"Epoch {epoch} "
                    f"nll: {model_loss.item():.3f} "
                    f"acc (best): {100*avg_stop.current_score:.1f}"
                    f" ({100*avg_stop.previous_best:.1f})"
                    f" adv (worst): {adv_stop.current_score:.2f}"
                    f" ({adv_stop.previous_best:.2f})"
                    f"{dro_str}"
                    f"{gpu_str}"
                ))
            # Stop based on step
            if stop_by_step and step >= optim_args.n_steps:
                break

    # Return best ERM and adv state dict
    return th.load(model_file), th.load(adv_model_file)


def get_args():
    experiment = Experiment("Adversarial DRO")
    # Experimental setting
    general_args = ArgumentGroup("General")
    general_args.add_argument("--random-seed", type=int, default=9878888)
    general_args.add_argument("--n-reruns", type=int, default=1,
                              help="Number of reruns (with different seeds)")
    general_args.add_argument("--force-save-name", type=str, default=None,
                              help="Force using a specific save name")
    general_args.add_argument("--no-cuda", action="store_true",
                              help="Force CPU")
    general_args.add_argument("--dry-run", action="store_true",
                              help="Dry run (smaller training and eval data)")
    general_args.add_argument("--exp-prefix", type=str, default="",
                              include_in_name=True)
    general_args.add_argument("--data-dir", type=str, default="datasets")
    general_args.add_argument("--test-only", action="store_true",
                              help="Just test")
    general_args.add_argument("--results-folder", type=str, default="results/",
                              help="Folder in which to save results")
    general_args.add_argument("--test-on-split", type=str, default="test",
                              help="Which split should we test on")
    general_args.add_argument("--train-log-interval", type=int, default=1)
    general_args.add_argument("--eval-on-domains", type=str, default=None,
                              nargs="*")
    general_args.add_argument("--eval-on-canonical-domains",
                              action="store_true",
                              help="Use the canonical subdomain splits")
    general_args.add_argument("--task", default="biased_SST_95",
                              include_in_name=True,
                              choices=list(task_list.keys()),)
    experiment.add_configuration(general_args)
    # Model specific arguments
    pdro_args.add_model_args(experiment)
    # Adversary specific argument
    pdro_args.add_adversary_args(experiment)
    # Optimization arguments
    pdro_args.add_optimization_args(experiment)
    # Group dro arguments
    pdro_args.add_group_dro_args(experiment)
    # Parse arguments
    experiment.parse_args()
    return experiment._configs_by_name, experiment.make_exp_name()


def parse_domain(domain_descriptor):
    """Given a string of the format [attr]=[value],... return a function
    that returns True iff all attributes [attr] have value [value]

    Args:
        domain_descriptor (str): List of attributes and values

    Returns:
        function: function to use for filtering
    """
    domain_attributes = {}
    if len(domain_descriptor) == 0:
        return lambda x: True
    for k_v in domain_descriptor.split(","):
        k, v = k_v.split("=")
        domain_attributes[k] = v

    def filtering(attr):
        for k, v in domain_attributes.items():
            if k not in attr or str(attr[k]) != v:
                return False
        return True

    return filtering


def eval_on_domains(model, task, eval_domain_filters, split="valid"):
    """Evaluate a model on a list of domains"""
    domain_scores = {}
    full_data = task.get_split(split)
    for domain, domain_filter in eval_domain_filters.items():
        # Select domain data
        domain_test_data = full_data.filtered(domain_filter)
        # Compute score on domain
        domain_scores[domain] = task.eval_model(model, data=domain_test_data)
    return domain_scores


def make_adversary(
    architecture: str,
    filename: str,
    input_shape: Tuple[int, ...],
    output_size: Tuple[int, ...],
    device: str = "gpu:1",
    ratio_model: bool = False,
) -> th.nn.Module:
    """Create the adversary

    Args:
        architecture (str): Architecture
        filename (str): Path to MLE model
        input_shape (Tuple[int, ...]): Shape of the inputs.
        output_size (Tuple[int, ...]): Shape of the outputs.
        device (str, optional): Device. Defaults to "gpu:1".
        ratio_model (bool, optional): TODO. Defaults to False.

    Returns:
        The adversary with the MLE parameters loaded in
    """
    if ratio_model:
        # In this case, the adversary models the ratio q / p directly.
        # It is a model that takes an input and returns a real number
        # log (q / p), unnormalized. If we are modeling the join distribution,
        # This returns a vector where row i corresponds
        # to log (q(x, i)/p(x, i))
        adv = build_model(architecture, input_shape, output_size=None)
        head = th.nn.Linear(adv.hidden_size, output_size)
        # We initialize the head at 0, so the starting log ratio will be 0
        # But this is not necessary after all
        # head.weight.data.zero_()
        # head.bias.data.zero_()
        adv = ModelWithHead(adv, head)
    else:
        adv = build_model(architecture, input_shape, output_size)
        # There is no classification head
        adv = ModelWithHead(adv)
    # Maybe load a pre-trained model
    if filename is not None:
        adv_state_dict = th.load(filename, map_location=device)
        adv.load_state_dict(adv_state_dict, strict=False)
    return adv.to(device)


def main():
    configs, exp_name = get_args()
    # Retrieve specific arguments
    general_args = configs["General"]
    model_args = configs["Model"]
    adv_args = configs["Adversary"]
    optim_args = configs["Optimization"]
    dro_args = configs["Group-DRO"]
    # decide whether to use cuda or not.
    if th.cuda.is_available() and not general_args.no_cuda:
        device = th.device("cuda:0")
    else:
        device = th.device("cpu")
    # Fix random seed
    np.random.seed(general_args.random_seed)
    th.random.manual_seed(general_args.random_seed+1)
    # th.backends.cudnn.enabled = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    # prepare task datasets
    input_format = model_args.input_format
    if input_format is None:
        input_format = model_args.architecture
    task, input_shape, output_size = prepare_task(
        general_args.task,
        path=general_args.data_dir,
        model_name=input_format
    )
    # In this case the adversary is a proper generative model
    adv_output_size = None
    # If the adversary models x, y or x | y, we need to specify the number
    # of classes
    if adv_args.joint or adv_args.class_conditional:
        adv_output_size = task.n_classes
    elif adv_args.ratio_model:
        adv_output_size = 1
    # Create adversary
    adv = make_adversary(
        adv_args.adv_architecture,
        filename=adv_args.adv_filename,
        input_shape=input_shape,
        output_size=adv_output_size,
        device=device,
        ratio_model=adv_args.ratio_model,
    )
    # Pre-compute baseline LM scores
    if not adv_args.pdro or adv_args.ratio_model or adv_args.non_param:
        # This is a hack: the "baseline" log probilities are not used in
        # this scenario
        all_log_probs = th.zeros(len(task.train_data))
    else:
        # Pre-compute language modeling scores
        adv_type = "_cc" if adv_args.class_conditional else ""
        if adv_args.adv_filename is not None:
            adv_filename = os.path.basename(adv_args.adv_filename)
        else:
            adv_filename = adv_args.adv_architecture
        all_log_probs_filename = os.path.join(
            general_args.results_folder,
            f"lm_{adv_filename}{adv_type}_train",
        )
        # Pre-compute the log probabilities of the training samples
        all_log_probs = compute_dataset_log_probs(
            adv,
            task,
            "train",
            batch_size=optim_args.batch_size,
            max_tokens_per_batch=optim_args.max_tokens_per_batch,
            joint=adv_args.joint,
            class_conditional=adv_args.class_conditional,
            cached_filename=all_log_probs_filename,
        )
    # Add the baseline log p as an attribute to the train data
    # First initialize attributes (FIXME: this should be removed)
    if (not hasattr(task.train_data, "attributes") or
            task.train_data.attributes is None):
        task.train_data.attributes = [{} for _ in range(len(task.train_data))]
    # Sanity check
    if not len(all_log_probs) == len(task.train_data):
        raise ValueError(f"{len(all_log_probs)} != {len(task.train_data)}")
    for idx, log_p in enumerate(all_log_probs):
        task.train_data.attributes[idx]["log_p"] = log_p

    # Evaluation domains: these are the domains that will be used for
    # computing the robust loss. Each domain is defined by a function that
    # takes a sample and returns True if the sample is in domain, False
    # otherwise
    eval_domain_filters = None
    # Evaluate on canonical domains
    if general_args.eval_on_canonical_domains:
        general_args.eval_on_domains = task.canonical_domain_descriptors
    if general_args.eval_on_domains is not None:

        eval_domain_filters = {domain: parse_domain(domain)
                               for domain in general_args.eval_on_domains}
    # Set up for group DRO (a baseline where the groups are known)
    # Similarly, the training domains are defined as a boolean function on the
    # samples
    train_domain_filters = None
    valid_pseudo_domain_filters = None
    if dro_args.group_specifications is not None:
        # In this case, the groups are directly defined by the sample's
        # attributes
        train_domain_filters = {
            domain: parse_domain(domain)
            for domain in dro_args.group_specifications
        }
    elif dro_args.train_group_file is not None:
        # This is when the groups are defined by a file containing the
        # domain label of each sample
        train_group_labels = np.load(dro_args.train_group_file)
        valid_group_labels = np.load(dro_args.dev_group_file)
        # Groups on the training data
        groups = list(set(train_group_labels))
        train_idxs = np.arange(len(task.train_data))
        train_domain_filters = {
            str(group): train_idxs[train_group_labels == group]
            for group in groups
        }
        valid_idxs = np.arange(len(task.valid_data))
        valid_pseudo_domain_filters = {
            str(group): valid_idxs[valid_group_labels == group]
            for group in groups
        }
    elif dro_args.gold_groups:
        # This is when we are using the same domains as the evaluation
        # domains
        train_domain_filters = {k: v for k,
                                v in eval_domain_filters.items()}
    # Name of the file for saving
    if general_args.force_save_name is not None:
        save_name = general_args.force_save_name
    else:
        # (FIXME: this is a horrible solution)
        save_name = hashlib.md5(exp_name.encode("utf-8")).hexdigest()
    # Now evaluate on the test tasks for a number of runs
    for run_id in range(general_args.n_reruns):
        # Fix random seed
        np.random.seed(general_args.random_seed+run_id)
        th.random.manual_seed(general_args.random_seed+run_id+1)
        # Experiment name
        if general_args.n_reruns > 1:
            run_exp_name = f"{exp_name}_run_{run_id}"
            run_save_name = f"{save_name}_run_{run_id}"
        else:
            run_exp_name = exp_name
            run_save_name = save_name
        print(f"Saving experiment: {run_exp_name} to {save_name}")
        # prepare the model
        model = build_model(model_args.architecture, input_shape, None)
        head = task.create_compatible_head(model.hidden_size)
        model = ModelWithHead(model, head)
        # Load model weights as needed
        if model_args.load_model_from_file is not None:
            saved_model_state_dict = th.load(
                model_args.load_model_from_file,
                map_location=device,
            )
            model.load_state_dict(saved_model_state_dict)
        # Cast model to device
        model = model.to(device)
        # Reset adversary
        adv = make_adversary(
            adv_args.adv_architecture,
            filename=adv_args.adv_filename,
            input_shape=input_shape,
            output_size=adv_output_size,
            device=device,
            ratio_model=adv_args.ratio_model,
        )
        # Print all arguments
        print("=" * 100)
        for cfg_name, cfg in configs.items():
            print(cfg)
        print("="*100, flush=True)
        # Training
        if not general_args.test_only:
            best_erm_model_state_dict, best_adv_model_state_dict = train(
                model,
                adv,
                task,
                model_args,
                adv_args,
                optim_args,
                dro_args,
                train_log_interval=general_args.train_log_interval,
                device=device,
                exp_name=run_exp_name,
                save_name=run_save_name,
                figure_prefix="precisions",
                results_prefix=general_args.results_folder,
                eval_domain_filters=eval_domain_filters,
                train_domain_filters=train_domain_filters,
                valid_pseudo_domain_filters=valid_pseudo_domain_filters,
            )
        # Test
        test_split = general_args.test_on_split
        for stopping_strategy in ["erm", "adv", "robust"]:
            print(
                f"Testing best {stopping_strategy} model on task {task.name}",
                flush=True
            )
            # load the model
            save_file = f"{run_save_name}_{stopping_strategy}_model.pt"
            if stopping_strategy == "erm":
                save_file = f"{run_save_name}_model.pt"
            best_model_state_dict = th.load(os.path.join(
                general_args.results_folder, save_file))
            model.load_state_dict(best_model_state_dict)
            # Evaluate on general domain
            score = task.eval_model(model, data=test_split)
            print(f"Score: {score*100:.3f}", flush=True)
            # Evaluate on specific domains
            domain_scores = eval_on_domains(model, task, eval_domain_filters)
            for domain, domain_score in domain_scores.items():
                print(f"Score on domain {domain}: {100*domain_score:.3f}")
            # Save model predictions
            log_p, y_hat = task.predict_dataset(model, data=test_split)
            # Also save references
            split_data = task.get_split(test_split)
            y = np.asarray(split_data.get_labels())
            results_file = f"{run_save_name}_{test_split}_results_{stopping_strategy}.npz"  # noqa
            np.savez_compressed(
                os.path.join(general_args.results_folder, results_file),
                exp_name=exp_name,
                log_p=log_p.numpy(),
                y_hat=y_hat.numpy(),
                y=y,
                domain_scores=domain_scores,
                name=task.name,
            )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit()
