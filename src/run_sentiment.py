# run_sentiment.py
import pandas as pd
import os
import traceback
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings

# Suppress the warning about sequential GPU usage
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")

def run_sentiment_analysis(models, nicknames, text_cols, text_col_suffixes, 
                          output_dir, input_filename, output_filename, 
                          hf_home=None, batch_size=32, progress_callback=None):
    """
    Run sentiment analysis on text data using HuggingFace models
    
    Args:
        models: List of HuggingFace model names
        nicknames: Dict mapping model names to short nicknames
        text_cols: List of text column names to analyze
        text_col_suffixes: Dict mapping text columns to output suffixes
        output_dir: Directory containing input and output files
        input_filename: Name of input parquet file
        output_filename: Name of output parquet file
        hf_home: Optional HuggingFace cache directory
        batch_size: Batch size for processing (default: 32)
        progress_callback: Optional callback for progress updates
    
    Returns:
        DataFrame with sentiment analysis results or None if error
    """
    
    # Set HuggingFace cache if provided
    if hf_home:
        os.environ['HF_HOME'] = hf_home
        print(f"[INFO] HuggingFace cache set to: {hf_home}")
    
    # File paths
    input_path = os.path.join(output_dir, input_filename)
    output_path = os.path.join(output_dir, output_filename)
    
    # Display configuration
    print(f"\n[CONFIG] Sentiment Analysis Configuration:")
    print(f"  - Models: {len(models)} models")
    print(f"  - Text columns: {text_cols}")
    print(f"  - Input: {input_path}")
    print(f"  - Output: {output_path}")
    print(f"  - Batch size: {batch_size}")
    
    # Helper function to standardize sentiment labels
    def standardize_label(label):
        """Convert various label formats to standard: Negative/Neutral/Positive"""
        if not isinstance(label, str):
            return 'Other'
        label = label.lower()
        if label in ['negative', 'neg', 'label_0']:
            return 'Negative'
        elif label in ['neutral', 'neu', 'label_1']:
            return 'Neutral'
        elif label in ['positive', 'pos', 'label_2']:
            return 'Positive'
        else:
            return 'Other'
    
    # Step 1: Load data
    print(f"\n[STEP 1] Loading data...")
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return None
    
    try:
        df = pd.read_parquet(input_path)
        print(f"Loaded {len(df):,} rows")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None
    
    # Check required columns
    missing_cols = [col for col in text_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Step 2: Setup device
    print(f"\n[STEP 2] Setting up device...")
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    #print(f"Using {device_name}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  - GPU: {gpu_name}")
        print(f"  - Memory: {gpu_memory:.1f} GB")
    
    # Step 3: Load models
    print(f"\n[STEP 3] Loading sentiment models...")
    pipelines = {}
    
    for i, model_name in enumerate(models, 1):
        nickname = nicknames.get(model_name, model_name)
        print(f"\nLoading model {i}/{len(models)}: {nickname}")
        
        try:
            # Create pipeline
            pipe = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=device,
                truncation=True,
                padding=True,
                max_length=512
            )
            pipelines[model_name] = pipe
            print(f"Successfully loaded {nickname}")
            
        except Exception as e:
            print(f"Failed to load '{nickname}': {e}")
            continue
    
    if not pipelines:
        print("ERROR: No models loaded successfully!")
        return None
    
    print(f"\nAll models loaded successfully ({len(pipelines)}/{len(models)})")
    
    # Step 4: Process sentiment analysis
    print(f"\n[STEP 4] Running sentiment analysis...")
    results_df = df.copy()
    
    # Process each combination
    analysis_num = 1
    total_analyses = len(text_cols) * len(pipelines)
    
    for text_col in text_cols:
        suffix = text_col_suffixes.get(text_col, text_col)
        
        # Get texts and handle NaN
        texts = df[text_col].fillna('').astype(str).tolist()
        
        for model_name, pipe in pipelines.items():
            nickname = nicknames.get(model_name, model_name)
            col_prefix = f"sentiment_{nickname}_{suffix}"
            
            print(f"\n[Analysis {analysis_num}/{total_analyses}] {nickname} on {text_col}")
            
            try:
                # Process in batches with individual progress bar
                all_results = []
                
                # Calculate number of batches
                num_batches = (len(texts) + batch_size - 1) // batch_size
                
                # Progress bar for this specific analysis
                with tqdm(total=len(texts), 
                         desc=f"Processing {nickname}", 
                         unit="texts",
                         ncols=100,
                         leave=True) as pbar:
                    
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_results = pipe(batch_texts)
                        all_results.extend(batch_results)
                        
                        # Update progress
                        pbar.update(len(batch_texts))
                
                # Extract labels and scores
                labels = []
                scores = []
                
                for result in all_results:
                    # Handle both single dict and list of dicts
                    if isinstance(result, list):
                        result = result[0]  # Take top prediction
                    
                    label = standardize_label(result.get('label', 'Other'))
                    score = float(result.get('score', 0.0))
                    
                    labels.append(label)
                    scores.append(score)
                
                # Add columns to dataframe
                results_df[f"{col_prefix}_label"] = labels
                results_df[f"{col_prefix}_score"] = scores
                
                # Show results summary
                label_counts = pd.Series(labels).value_counts()
                print(f"Results: {label_counts.to_dict()}")
                
            except Exception as e:
                print(f"Error processing {nickname} on {text_col}: {e}")
                # Add error columns
                results_df[f"{col_prefix}_label"] = 'ERROR'
                results_df[f"{col_prefix}_score"] = 0.0
            
            analysis_num += 1
    
    # Step 5: Save results
    print(f"\n[STEP 5] Saving results...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        # Summary
        sentiment_cols = [col for col in results_df.columns if 'sentiment_' in col]
        print(f"\n--- Summary ---")
        print(f"Total rows processed: {len(results_df):,}")
        print(f"Sentiment columns added: {len(sentiment_cols)}")
        
        return results_df
        
    except Exception as e:
        print(f"ERROR saving results: {e}")
        traceback.print_exc()
        return None


# Command-line interface (for running from terminal)
if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run sentiment analysis")
    parser.add_argument("--models", nargs='+', required=True, help="Model names")
    parser.add_argument("--nicknames-json", type=str, required=True, help="JSON mapping")
    parser.add_argument("--text-cols", nargs='+', required=True, help="Text columns")
    parser.add_argument("--text-col-suffixes-json", type=str, required=True, help="Suffixes JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--input-filename", type=str, required=True, help="Input file")
    parser.add_argument("--output-filename", type=str, required=True, help="Output file")
    parser.add_argument("--hf-home", type=str, help="HuggingFace cache directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    nicknames = json.loads(args.nicknames_json)
    text_col_suffixes = json.loads(args.text_col_suffixes_json)
    
    print("\nStarting sentiment analysis from command line...")
    
    run_sentiment_analysis(
        models=args.models,
        nicknames=nicknames,
        text_cols=args.text_cols,
        text_col_suffixes=text_col_suffixes,
        output_dir=args.output_dir,
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        hf_home=args.hf_home,
        batch_size=args.batch_size
    )