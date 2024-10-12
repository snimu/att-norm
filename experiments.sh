# go through all settings, at 46M parameters
python main.py -c --logfile results1.csv -w --wandb_project att-norm.1 --qk_activ gelu none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid tanh none --post_attn_norm rms_norm none --num_runs 5 --seed 12345

# go through the baseline and the best three other settings at 240M parameters
python main.py -c --logfile results2.csv -w --wandb_project att-norm.2 --qk_activ none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid --post_attn_norm rms_norm none --num_runs 5 --seed 1 --depth 21 --width 1024 --num_heads 16 --gpu_capacity_scalar 1.9  # done on H100

# Now even larger scale
python main.py -c --logfile results3.csv -w --wandb_project att-norm.3 --qk_activ gelu none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid --post_attn_norm none --num_runs 5 --seed 1 --depth 35 --width 1664 --num_heads 26 --gpu_capacity_scalar 1.9  # done on H100

# ? have to change settings
python main.py -c --logfile results3.csv -w --wandb_project att-norm.3 --qk_activ gelu none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid --post_attn_norm none --num_runs 5 --seed 1 --depth 43 --width 2048 --num_heads 32 --gpu_capacity_scalar 1.9  # done on H100