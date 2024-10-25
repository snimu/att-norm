# Logbook

The logbook of my experiments.

## 2024-10-10

First experiments.

### 2024-10-10: dipping my toes in

- Created repo
- Implemented all the attention stuff and settings
- Tested those by going through all the new setting one by one --> bugfixes

Tested a bunch of combinations of settings for two steps each, just to get an initial feeling for the results.

- `tanh` and `none` are terrible settings for `attn_activ`
    - omg that's because I don't apply positional encodings there!!!
    - to change that, add the attn-mask, then multiply with it

Plan:

- Fix that
- Keep the tests discussed below running (they have already started and I don't want to interrupt them)
- Quick test of `attn_activ=tanh` and `attn_activ=none`
- If that is even trivially promising, test them properly (like in the tests below)

Update: there was another stupid mistake: `torch.norm` didn't use the `dim=-1` argument, so it was very different from what I wanted. Fixed that. While at it, also fixed the attention-mask thing.

The small experiments now show that `tanh` and `none` reduce the losses (as opposed to before), so I will restart the experiments with the bugfixes and include these two.


### 2024-10-10: Ablations

Performed the ablations shown in `experiments.sh` (first run):

```bash
python main.py -c --logfile results1.csv -w --wandb_project att-norm.1 --qk_activ gelu none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid tanh none --post_attn_norm rms_norm none --num_runs 5 --seed 12345
```

**Results**

First off, always keeping all other settings except the `attn_activ` at `none`:

- `attn_activ`: `sigmoid` and `softmax` are the best (seem very similar)
- `qk_norm` (between those two): 
  - `none`: is best
  - `fro_norm`: consistently worse by ~$0.2$ in loss value
  - `rms_norm`: consistently worse by ~$0.2$ in loss value for `sigmoid`, but for `softmax` it makes no difference
- `post_attn_norm` (between `softmax` and `sigmoid`):
  - `none`: is best
  - `rms_norm`: consistently worse by between $1.5$ and $2$ in loss value

Now, trying some specific combinations of settings (for `attn_activ=sigmoid` and `attn_activ=softmax`):

- Ablation 1
  - Settings:
    - `attn_activ`: `sigmoid` | `softmax`
    - `qk_norm`: `rms_norm`
    - `qk_activ`: `gelu` | `none`
  - Results:
    - `softmax` can handle `rms_norm` (it makes no difference); `sigmoid` cannot (it degrades performance a lot)
    - The `gelu` activation makes no real difference
- Ablation 2
  - Settings:
    - `attn_activ`: `sigmoid` | `softmax`
    - `qk_norm`: `fro_norm` | `none`
    - `qk_activ`: `gelu` | `none`
  - Results:
    - The `fro_norm` simply destroys performance in all cases.
    - For `qk_norm=none`, the `gelu` activation actually seems to worsen performance a bit.

## 2024-10-11

First off, holy fuck F.scaled_dot_product_attention is so efficient! Using my self-cooked, manually implemented attention version takes sooo much memory it's crazy. Have to run the script with --gpu_capacity_scalar 1.5 for it to work *on an H100*!

I'm beginning to suspect that the norms are a problem not because they are bad, but because they change training dynamics so much and I haven't tuned the hyperparameters for them at all. Why do I think that? The microbatch-number per update is dynamically adjusted depending on the grad norm; so if something is wrong with the batch norm, the microbatch-number is very large. We will see few updates before 1 epoch is done; and that is exactly what is happening.

So maybe I should tune the small model for the norms next?

## 2024-10-12

### 2024-10-12: Results of yesterday's ablations

- `post_attn_norm` should just be `none`
- `qk_norm`: `none` > `rms_norm` > `fro_norm` but the difference is small enough that proper tuning might fix it?


## 2024-10-13

- Still no real difference between `sigmoid` and `softmax` if all others are `none`
- `qk_activ`: no difference between `gelu` and `none`
- `qk_norm`: `none` >> `rms_norm` > `fro_norm` for both `sigmoid` and `softmax`

Might the problem be the positional encodings? They are applied with the attention mask instead on Q and K directly! I should try this with a different repo. Probably [llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py).


## 2024-10-25

Started training with llm.c by karpathy.

Early results:

- sigmoid is significantly worse than softmax
  - Why is this not the case for hlb-gpt?
  - I'm guessing that it's because of the positional encodings
  - In that case, scaling the attention logits should help
- Scaling attention logits *does* help, but similarly for softmax and sigmoid
  - Sigmoid still isn't as good as softmax without the logit scaling
  - I should try the actual positional encodings next (in addition to RoPE!)

Result with hlb-gpt-style positional encodings:

- Makes basically zero difference.

