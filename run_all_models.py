import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

def get_sp500_it_tickers():
    """
    Reads S&P 500 IT sector tickers from Wikipedia or uses a fallback list.
    """
    try:
        print("Fetching S&P 500 IT sector tickers for test set...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})
        sp500_df = tables[0]
        it_tickers = sp500_df[sp500_df['GICS Sector'] == 'Information Technology']['Symbol'].tolist()
        it_tickers = [ticker.replace('.', '-') for ticker in it_tickers]
        print(f"Found {len(it_tickers)} tickers in the IT sector.")
        return it_tickers
    except Exception as e:
        print(f"Could not fetch tickers from Wikipedia, using a fallback list. Error: {e}")
        return ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'ACN', 'ORCL', 'ADBE', 'CRM', 'INTC', 'CSCO']

def get_test_tickers(test_size=0.2, random_state=42):
    """
    Gets the full ticker list and returns the test split.
    """
    tickers = get_sp500_it_tickers()
    _, test_tickers = train_test_split(tickers, test_size=test_size, random_state=random_state)
    return test_tickers

def parse_script_output(output):
    """
    Parses the machine-readable block from a script's captured output.
    """
    results = {}
    in_output_block = False
    for line in output.splitlines():
        if line.strip() == '--- SCRIPT_OUTPUT ---':
            in_output_block = True
            continue
        if line.strip() == '--- END_SCRIPT_OUTPUT ---':
            break
        
        if in_output_block and '=' in line:
            key, value = line.strip().split('=', 1)
            results[key] = value
            
    # Post-process loss history from string to list of floats
    if 'LOSS_HISTORY' in results:
        loss_str = results['LOSS_HISTORY'].strip('[]')
        if loss_str:
            results['LOSS_HISTORY'] = [float(x) for x in loss_str.split(',')]
        else:
            results['LOSS_HISTORY'] = []
            
    return results

def plot_loss_comparison(results, timestamp):
    """
    Plots the training loss progression for all models on a single graph.
    """
    plt.figure(figsize=(12, 8))
    for result in results:
        model_name = result.get('MODEL_NAME', 'Unknown Model')
        loss_history = result.get('LOSS_HISTORY', [])
        if loss_history:
            plt.plot(loss_history, label=f"{model_name} Loss")
            
    plt.title('Training Loss Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f"loss_comparison_plot_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"\nLoss comparison plot saved to: {plot_filename}")
    plt.close()

def log_experiment_results(results, timestamp):
    """
    Logs the final metrics for all models to a dedicated experiment log file.
    """
    log_filename = "experiment_log.log"
    
    try:
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(f"--- Experiment Run: {timestamp} ---\n")
            f.write("Timestamp,Model,TrainingTime(s),FinalLoss,TestRMSE\n")
            for result in results:
                model = result.get('MODEL_NAME', 'N/A')
                train_time = result.get('TRAINING_TIME', 'N/A')
                final_loss = result.get('FINAL_LOSS', 'N/A')
                test_rmse = result.get('TEST_RMSE', 'N/A')
                f.write(f"{timestamp},{model},{train_time},{final_loss},{test_rmse}\n")
            f.write("---" * 10 + "\n\n")
        print(f"Experiment results logged to: {log_filename}")
    except IOError as e:
        print(f"Error logging experiment results: {e}")

def execute_script_in_realtime(command):
    """
    Executes a script using Popen to show output in real-time
    and captures it for parsing. Returns the full output and exit code.
    """
    full_output = []
    try:
        # Popen starts the process and continues
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            encoding='utf-8',
            bufsize=1 # Line-buffered
        )

        # Read output line by line in real-time
        while True:
            line = process.stdout.readline()
            if not line:
                break
            # Print to console and also store it
            sys.stdout.write(line)
            full_output.append(line)
        
        # Wait for the process to complete and get the exit code
        process.wait()
        return "".join(full_output), process.returncode

    except FileNotFoundError:
        return f"--- ERROR: Python interpreter '{command[0]}' not found. ---", -1
    except Exception as e:
        return f"An unexpected error occurred: {e}", -1


def main():
    """
    Main function to run the experiment, collect results, and generate outputs.
    """
    print("--- Starting Coordinated Model Experiment Runner ---")
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    
    # 1. Get tickers and select 2 for validation plots
    test_tickers = get_test_tickers()
    if len(test_tickers) < 2:
        print("Error: Not enough test tickers available.")
        return
    random.seed() 
    validation_tickers = random.sample(test_tickers, 2)
    print(f"\nTickers selected for validation plots: {', '.join(validation_tickers)}")

    # 2. Define model scripts and storage for results
    scripts_to_run = [
        'LSTM stock app.py',
        'RNN stock app.py',
        'Transformer stock app.py'
    ]
    all_results = []

    # 3. Run each script and collect its output
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"\n--- ERROR: Script '{script}' not found. Skipping. ---")
            continue

        print(f"\n-=-=-=-= Running {script} =-=-=-=-=")
        command = [sys.executable, script] + validation_tickers
        
        # Execute using the new real-time method
        script_output, exit_code = execute_script_in_realtime(command)
        
        if exit_code == 0:
            parsed_data = parse_script_output(script_output)
            if parsed_data:
                all_results.append(parsed_data)
                print(f"--- Successfully parsed results for {parsed_data.get('MODEL_NAME')}")
            else:
                print(f"--- WARNING: Could not parse results for {script}. Check script output. ---")
        else:
            print(f"--- ERROR running {script} (Exit Code: {exit_code}). Stopping runner. ---")
            break # Stop the entire process if one script fails
    
    # 4. Generate final outputs if all scripts ran successfully
    if len(all_results) == len(scripts_to_run):
        print("\n--- All models ran. Generating final outputs. ---")
        plot_loss_comparison(all_results, timestamp)
        log_experiment_results(all_results, timestamp)
    else:
        print("\n--- Runner did not complete all models. Skipping final output generation. ---")

if __name__ == "__main__":
    main()