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