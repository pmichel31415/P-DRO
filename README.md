# Parametric Distributionally Robust Optimization

This repository contains code for implementing experiments in the ICLR 2021 paper "Modeling the Second Player in Distributionally Robust Optimization".

## How to run this code

After checking out the repository, you need to install the dependencies, for instance in a virtual environment:

```bash
virtualenv --system-site-packages env
source env/bin/activate
pip install -r requirements.txt
```

Now you should be ready to go. Here is an example run on BiasedSST (see paper for details).

Normally you should first pre-train the MLE of the data distribution (`q_{\psi_0}` in the paper). However, to get you started you can just download our [pre-trained model](https://github.com/pmichel31415/P-DRO/releases/download/1.0/biased_SST_95_gen_LM_small_transformer_generative_wikitext103_model.pt).

```bash
# Download pre-trained models
mkdir pretrained_models/
wget https://github.com/pmichel31415/P-DRO/releases/download/1.0/biased_SST_95_gen_LM_small_transformer_generative_wikitext103_model.pt -P pretrained_models/
# Create results folder
mkdir results/
```

Now you can run P-DRO on BiasedSST using the following command:

```bash
python pdro_main.py \
    --force-save-name biased_sst_p_dro \
    --n-reruns 5 \
    --task biased_SST_95 \
    --architecture bilstm \
    --input-format bert-base-uncased \
    --lr-scheduler linear_decay \
    --n-epochs 50 \
    --valid-interval epoch \
    --optimizer adamw \
    --lr 2e-5 \
    --batch-size 64 \
    --max-tokens-per-batch 2500 \
    --eval-on-domains biased=True,label=0 biased=True,label=1 biased=False,label=0 biased=False,label=1 \
    --pdro \
    --adv-architecture small_transformer_generative \
    --adv-filename pretrained_models/biased_SST_95_gen_LM_small_transformer_generative_wikitext103_model.pt \
    --filter-advs-by reverse_kl \
    --adv-threshold 2.302585 \
    --adv-optimizer sgd  \
    --adv-lr 1e-4 \
    --adv-obj exp_kl \
    --joint \
    --tau 0.01 \
    --norm-k-adv 5
```

Note the parameters relevant to P-DRO:

- `--pdro`: use P-DRO (deactivate this for ERM)
- `--adv-architecture small_transformer_generative`: architecture for the adversary. You can find more architectures in `src/models/architectures.py`.
- `--filter-advs-by reverse_kl`: filter validation adversary by reverse KL
- `--adv-threshold 2.302585`: reject adversaries with reverse KL to the data distribution > `log(10)` on the dev set
- `--adv-optimizer sgd`: train the adversary with regular SGD
- `--adv-lr 1e-4`: train the adversary with learning rate 1r-4 (this is `\lambda` in the paper)
- `--adv-obj exp_kl`: this refers to L_adv as described in the paper
- `--joint`: this indicates that the adversary will model the joint distribution `(x, y)` (instead of just x)
- `--tau 0.01`: Temperature for the adversary's loss
- `--norm-k-adv`: This is the size of the window for computing the normalizer in the adversary's loss (`K` in the paper)
