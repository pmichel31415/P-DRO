#!/usr/bin/env python3
from src.optim import optimizers, lr_schedulers
from src.models import architecture_list
from src.configuration import ArgumentGroup


def add_model_args(experiment):
    model_args = ArgumentGroup("Model")
    model_args.add_argument('--architecture', type=str,
                            default="bert-base-uncased",
                            include_in_name=True,
                            choices=list(architecture_list.keys()))
    model_args.add_argument('--input-format', type=str, default=None,
                            choices=[None, "bert-base-uncased", "gpt2"],
                            help="Format (tok+vocabulary) for text input. "
                            "If None: decided based on the architecture.")
    model_args.add_argument("--hidden-dropout-prob", type=float, default=.5)
    model_args.add_argument("--input-dropout-prob", type=float, default=.2)
    model_args.add_argument("--load-model-from-file", default=None, type=str)
    experiment.add_configuration(model_args)


def add_adversary_args(experiment):
    adv_args = ArgumentGroup("Adversary")
    adv_args.add_argument("--pdro", action="store_true",
                          help="Train with Adv-DRO", include_in_name=True)
    adv_args.add_argument("--adv-architecture",
                          type=str, default="ff_lm",
                          choices=list(architecture_list.keys()),
                          include_in_name="pdro")
    adv_args.add_argument("--adv-filename",
                          type=str, default=None,
                          include_in_name="pdro")
    adv_args.add_argument("--adv-optimizer", type=str, default=None,
                          choices=list(optimizers.keys()),
                          include_in_name="pdro")
    adv_args.add_argument("--ewc-penalty", type=float, default=0)
    adv_args.add_argument("--renorm-ratios", action="store_true")
    adv_args.add_argument("--joint", action="store_true",
                          include_in_name="pdro")
    adv_args.add_argument("--class-conditional", action="store_true",
                          include_in_name="pdro")
    adv_args.add_argument("--alpha", type=float, default=1.0,
                          include_in_name="pdro")
    adv_args.add_argument("--beta", type=float, default=1.0,
                          include_in_name="pdro")
    adv_args.add_argument("--ratio-model", action="store_true",
                          help="Model the ratio directly",
                          include_in_name="pdro")
    adv_args.add_argument("--self-norm-lambda", type=float, default=0,
                          help="self normalization penalty",
                          include_in_name="ratio_model")
    adv_args.add_argument("--adv-obj", type=str, default="exp_kl",
                          include_in_name="pdro")
    adv_args.add_argument("--tau", type=float, default=1.0,
                          include_in_name="pdro", help="Temperature")
    adv_args.add_argument("--adv-on-acc", action="store_true",
                          help="Train adversary to maximize error "
                          "rate (not loss)",
                          include_in_name="pdro",)
    adv_args.add_argument("--norm-k-model", type=float, default=None,
                          include_in_name="pdro")
    adv_args.add_argument("--norm-k-adv", type=float, default=None,
                          include_in_name="pdro")
    adv_args.add_argument("--norm-model-only", action="store_true",
                          include_in_name="pdro")
    adv_args.add_argument("--norm-adv-only", action="store_true",
                          include_in_name="pdro")
    adv_args.add_argument("--adv-lr", type=float, default=None,
                          include_in_name="pdro")
    adv_args.add_argument("--adv-valid-on-acc", action="store_true",
                          help="Adversarial validation using accuracy "
                          "rather than loss",)
    adv_args.add_argument("--adv-mom", type=float, default=0,
                          include_in_name="pdro")
    adv_args.add_argument("--non-param", action="store_true",
                          include_in_name="pdro")
    adv_args.add_argument("--kappa", type=float, default=None,
                          include_in_name="non_param")
    adv_args.add_argument("--clip-grad-adv", type=float, default=None,
                          include_in_name="pdro")
    adv_args.add_argument("--adv-update-every", type=int, default=1)
    adv_args.add_argument("--filter-advs-by", type=str, default="none",
                          choices=["none", "reverse_kl", "alpha_coverage"])
    adv_args.add_argument("--adv-threshold", type=float, default=10000000)
    experiment.add_configuration(adv_args)


def add_optimization_args(experiment):
    optim_args = ArgumentGroup("Optimization")
    optim_args.add_argument("--optimizer", type=str, default="sgd",
                            choices=list(optimizers.keys()))
    optim_args.add_argument("--lr-scheduler", type=str, default="constant",
                            choices=list(lr_schedulers.keys()))
    optim_args.add_argument("--n-steps", type=int, default=500)
    optim_args.add_argument("--update-every", type=int, default=1)
    optim_args.add_argument("--lm-update-every", type=int, default=1)
    optim_args.add_argument("--n-epochs", type=int, default=None)
    optim_args.add_argument("--lr", type=float, default=1e-1)
    optim_args.add_argument("--weight-decay", type=float, default=0)
    optim_args.add_argument("--batch-size", type=int, default=32)
    optim_args.add_argument("--num-workers", type=int, default=1)
    optim_args.add_argument("--max-tokens-per-batch", type=int, default=None)
    optim_args.add_argument("--test-batch-size", type=int, default=32)
    optim_args.add_argument("--clip-grad", type=float, default=10.0)
    optim_args.add_argument("--valid-interval", default=250)
    optim_args.add_argument("--l2-reg", type=float, default=0,
                            include_in_name=True)
    experiment.add_configuration(optim_args)


def add_group_dro_args(experiment):
    dro_args = ArgumentGroup("Group-DRO")
    dro_args.add_argument("--group-dro", action="store_true",
                          help="Train with Group DRO",
                          include_in_name=True)
    dro_args.add_argument("--gold-groups", action="store_true",
                          help="Train with Group DRO on gold domains",
                          include_in_name="group_dro")
    dro_args.add_argument("--group-specifications",
                          type=str, default=None, nargs="*",
                          include_in_name="group_dro")
    dro_args.add_argument("--train-group-file", type=str, default=None,
                          include_in_name="group_dro")
    dro_args.add_argument("--dev-group-file", type=str, default=None,
                          include_in_name="group_dro")
    dro_args.add_argument("--eta-q", type=float, default=0.1,
                          include_in_name="group_dro")
    dro_args.add_argument("--baseline", type=float, default=None,
                          nargs="*", include_in_name="group_dro")
    experiment.add_configuration(dro_args)
