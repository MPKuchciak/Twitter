# run_sentiment.py
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm.auto import tqdm
import gc
import os
import traceback
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Optional: Suppress TensorFlow warnings ---
# print("Setting TensorFlow log level to ERROR...")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning, module='tf_keras')
# print("TensorFlow warning suppression attempted.")
# ----------------------------------------------


# === Configuration (Mirror relevant parts from Notebook Section 12.0) ===
SENTIMENT_MODELS_TO_RUN = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "ProsusAI/finbert"
]
SENTIMENT_MODEL_NICKNAMES = {
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "cardiffnlp",
    "ProsusAI/finbert": "finbert"
}
TEXT_COLUMNS_TO_ANALYZE = ['text_clean_en', 'text_clean_en_demojized']
TEXT_COLUMN_SUFFIXES = { 'text_clean_en': 'emoji', 'text_clean_en_demojized': 'demoji' }

# --- File Paths (Make sure these match notebook setup) ---
output_dir = '../data/05.final'
sentiment_input_filename = 'df_sentiment_input_CLEANED.parquet' # Input file created by Notebook Sec 11
sentiment_output_filename = 'df_sentiment_results.parquet' # Output file from this script

input_parquet_path = os.path.join(output_dir, sentiment_input_filename)
output_parquet_path = os.path.join(output_dir, sentiment_output_filename)

# === Helper Functions (Copy from Notebook if needed, e.g., standardize_label) ===
def standardize_label(lbl_input): 
    if not isinstance(lbl_input, str): return 'Other'
    s = lbl_input.lower()
    if s in ['negative', 'neg', 'label_0']: return 'Negative' 
    if s in ['neutral', 'neu', 'label_1']: return 'Neutral'  
    if s in ['positive', 'pos', 'label_2']: return 'Positive'
    if s == 'error_processing': return 'ERROR' 
    if s == 'error_no_score': return 'ERROR' 
    return 'Other'

# === Main Sentiment Analysis Logic ===
def run_analysis():
    print(f"Loading input data from: {input_parquet_path}")
    if not os.path.exists(input_parquet_path):
        print(f"ERROR: Input file not found: {input_parquet_path}")
        return None
    
    try:
        df_data = pd.read_parquet(input_parquet_path)
        print(f"Loaded {len(df_data)} rows.")
    except Exception as e:
        print(f"ERROR loading input parquet: {e}")
        return None

    # Check if required text columns exist
    missing_cols = [col for col in TEXT_COLUMNS_TO_ANALYZE if col not in df_data.columns]
    if missing_cols:
        print(f"ERROR: Input data missing required text columns: {missing_cols}")
        return None

    # -- 12.A Logic (Manual Loading - known to work in .py) --
    print("\n--- Setting up Sentiment Analysis Pipelines (Manual Loading) ---")
    sentiment_pipelines = {}
    device_num = 0 if torch.cuda.is_available() else -1
    effective_device = torch.device(f"cuda:{device_num}") if device_num == 0 else torch.device("cpu")
    print(f"Using device: {effective_device}")

    for model_name_hf in SENTIMENT_MODELS_TO_RUN:
        print(f"\nLoading components for: '{model_name_hf}'")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
            model_pt = AutoModelForSequenceClassification.from_pretrained(model_name_hf)
            model_pt.to(effective_device)
            model_pt.eval()
            pipe = pipeline("sentiment-analysis", model=model_pt, tokenizer=tokenizer, device=device_num, top_k=None)
            sentiment_pipelines[model_name_hf] = pipe
            print(f" -> Pipeline created for '{model_name_hf}'.")
        except Exception as e:
            print(f"ERROR setting up pipeline for '{model_name_hf}': {e}")
            traceback.print_exc()
            
    if not sentiment_pipelines:
        print("ERROR: No pipelines were loaded successfully. Aborting.")
        return None
    print(f"Successfully created {len(sentiment_pipelines)} pipeline(s).")
    print("-" * 50)

    # -- 12.B Logic (Applying Pipelines) --
    print("\n--- Applying Sentiment Analysis Pipelines ---")
    df_to_process = df_data # Process the whole loaded dataframe
    print(f"Processing {len(df_to_process)} tweets.")
    
    # --- Use Reduced Batch Size ---
    batch_size_to_use = 32 if device_num == 0 else 32 
    print(f"Using batch size: {batch_size_to_use}")

    for text_col_to_analyze in TEXT_COLUMNS_TO_ANALYZE:
        text_col_suffix = TEXT_COLUMN_SUFFIXES.get(text_col_to_analyze, text_col_to_analyze)
        print(f"\nProcessing Text Column: '{text_col_to_analyze}'")

        texts_to_analyze_list = df_to_process[text_col_to_analyze].fillna('').astype(str).tolist()

        for model_identifier, sentiment_pipeline_obj in sentiment_pipelines.items():
            model_nickname = SENTIMENT_MODEL_NICKNAMES.get(model_identifier, 'unknown_model')
            base_col_name = f"sentiment_{model_nickname}_{text_col_suffix}"
            label_col_name = f"{base_col_name}_label"
            score_col_name = f"{base_col_name}_score_best"
            score_neg_col_name = f"{base_col_name}_score_neg"
            score_neu_col_name = f"{base_col_name}_score_neu"
            score_pos_col_name = f"{base_col_name}_score_pos"

            print(f"  Running pipeline: '{model_identifier}'...")
            pipeline_results = []
            try:
                # Use pipeline directly on list with batching handled internally by pipeline
                # Setting batch_size in pipeline call might be more efficient if supported
                 pipeline_results = sentiment_pipeline_obj(texts_to_analyze_list, 
                                                           truncation=True, padding=True, max_length=512, 
                                                           batch_size=batch_size_to_use) # Pass batch_size here
            except Exception as e:
                 print(f"\nERROR: Pipeline application failed for {model_nickname}/{text_col_suffix}: {e}")
                 error_placeholder = [{'label': 'ERROR_PROCESSING', 'score': 0.0}]
                 pipeline_results = [error_placeholder] * len(texts_to_analyze_list) # Fill with errors

            print(f"  Finished '{model_identifier}'. Processing {len(pipeline_results)} results...")

            if len(pipeline_results) == len(df_to_process.index):
                try:
                    # ... which extracts labels, best_score, score_neg, score_neu, score_pos ...
                    parsed_labels, parsed_best_scores, parsed_scores_neg, parsed_scores_neu, parsed_scores_pos = [], [], [], [], []
                    for res_list_for_one_tweet in pipeline_results:
                        # ... (parsing logic) ...
                        current_tweet_scores = {}
                        if isinstance(res_list_for_one_tweet, list):
                           for r_dict in res_list_for_one_tweet:
                                if isinstance(r_dict, dict) and 'label' in r_dict and 'score' in r_dict:
                                   current_tweet_scores[r_dict['label'].lower()] = float(r_dict['score'])
                        
                        parsed_scores_neg.append(current_tweet_scores.get('negative', current_tweet_scores.get('label_0', 0.0)))
                        parsed_scores_neu.append(current_tweet_scores.get('neutral', current_tweet_scores.get('label_1', 0.0)))
                        parsed_scores_pos.append(current_tweet_scores.get('positive', current_tweet_scores.get('label_2', 0.0)))
                        
                        best_label_str, best_score_val = 'ERROR_NO_SCORE', 0.0
                        if 'error_processing' in current_tweet_scores: best_label_str = 'ERROR_PROCESSING'
                        elif current_tweet_scores: best_label_str = max(current_tweet_scores, key=current_tweet_scores.get)
                        
                        parsed_labels.append(standardize_label(best_label_str))
                        parsed_best_scores.append(current_tweet_scores.get(best_label_str, 0.0))

                    df_to_process[label_col_name] = parsed_labels
                    df_to_process[score_col_name] = parsed_best_scores
                    df_to_process[score_neg_col_name] = parsed_scores_neg
                    df_to_process[score_neu_col_name] = parsed_scores_neu
                    df_to_process[score_pos_col_name] = parsed_scores_pos
                    print(f"  -> Columns added for '{model_nickname}_{text_col_suffix}'.")
                except Exception as e:
                    print(f"ERROR assigning results for {model_nickname}_{text_col_suffix}: {e}")
                    traceback.print_exc()
            else:
                 print(f"ERROR: Length mismatch for {model_nickname}_{text_col_suffix}!")
            print("-" * 20) # End processing for one model/text combo

    # --- Save Results ---
    print("\n--- Saving results ---")
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True) 
        df_to_process.to_parquet(output_parquet_path, index=False)
        print(f"Successfully saved results with sentiment columns to: {output_parquet_path}")
        return df_to_process # Optionally return df if called from elsewhere
    except Exception as e:
        print(f"ERROR saving results to parquet: {e}")
        return None

# --- Script Entry Point ---
if __name__ == '__main__':
    # This guard is crucial for multiprocessing compatibility on Windows 
    print("Running sentiment analysis script...")
    resulting_df = run_analysis()
    if resulting_df is not None:
        print("Script finished successfully.")
    else:
        print("Script finished with errors.")