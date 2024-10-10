# go through all settings, at 46M parameters
python main.py -c --logfile results1.csv -w --wandb_project att-norm.1 --qk_activ gelu none --qk_norm fro_norm rms_norm none --attn_activ softmax sigmoid --post_attn_norm rms_norm none --num_runs 5 --seed 12345

# go through the baseline and the best three other settings at 240M, 773M, and 1.3B parameters