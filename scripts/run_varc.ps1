# Run VARC experiments (Exp 3 or 4) using the conda GPU environment
& "C:\Users\14544\.conda\envs\torch_gpu_py311\python" -m experiments.run_all --exp $args[0] @args[1..$args.Length]
