import optuna
import subprocess
import json

def objective(trial):
    # Define the hyperparameters to tune
    k = trial.suggest_int('k', 5, 50)
    lr = float(f"{trial.suggest_loguniform('lr', 1e-4, 1e-3):.2g}")
    delta = float(f"{trial.suggest_loguniform('delta', 1e-6, 1e-4):.2g}")
    lanczos_momentum = float(f"{trial.suggest_uniform('lanczos_momentum', 0, 1):.2g}")

    # Fixed hyperparameters
    batch_size = 8
    accumulation_steps = 8
    subsample = 0.25

    # Construct the command to run the script
    command = [
        'python', 'gpt2_hessian_gpu.py',
        '--batch_size', str(batch_size),
        '--accumulation_steps', str(accumulation_steps),
        '--k', str(k),
        '--subsample', str(subsample),
        '--lr', str(lr),
        '--delta', str(delta),
        '--lanczos_momentum', str(lanczos_momentum)
    ]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check if the process was successful
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        return float('inf')  # Return a high value indicating failure

    # Parse the output to get the metric value (e.g., loss)
    # This assumes that the script prints the loss in the final line of stdout
    try:
        last_line = result.stdout.strip().split('\n')[-1]
        loss = float(last_line)
    except Exception as e:
        print(f'Run Failed')
        print(f"Failed to parse the output: {e}")
        return float('inf')  # Return a high value indicating failure

    return loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print the best trial
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    # Save the study results
    with open('best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
