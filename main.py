from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler
import torch
import os
from nanoid import generate
import csv
import pandas as pd
from tqdm import tqdm
import gc

OUTPUT_DIR = "output"
BATCH_SIZE = 1
CHECKPOINT_INTERVAL = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.bfloat16,  # Keep float16 for speed
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing(1)
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = pipe.to("mps")
    
    pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    return pipe

def process_batch(pipe, rows, columns):
    results = []
    for row in rows:
        title = row["article_title"]
        # Enhanced prompt for better results
        prompt = f"A professional, text-free book cover for '{title}', featuring rich colors, a clear focal point, and no text or typography."

        
        try:
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=11,     # Balanced steps
                    guidance_scale=2,         # Increased for better adherence to prompt
                    width=1880,
                    height=240,
                    num_images_per_prompt=1,
                ).images[0]
            
            filename = generate() + ".png"
            image.save(os.path.join(OUTPUT_DIR, filename))
            
            data = dict(row)
            data["filename"] = filename
            results.append(data)
            
        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
            continue
            
    return results

def main():
    pipe = initialize_pipeline()
    
    df = pd.read_csv("data.csv")
    columns = list(df.columns) + ["filename"]
    
    if not os.path.exists("completed.csv"):
        pd.DataFrame(columns=columns).to_csv("completed.csv", index=False)
    
    completed_df = pd.DataFrame(columns=columns)
    
    with tqdm(total=len(df)) as pbar:
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            results = process_batch(pipe, batch.to_dict('records'), columns)
            
            completed_df = pd.concat([
                completed_df,
                pd.DataFrame(results)
            ], ignore_index=True)
            
            if i % CHECKPOINT_INTERVAL == 0:
                completed_df.to_csv("completed.csv", mode='a', header=False, index=False)
                completed_df = pd.DataFrame(columns=columns)
            
            df.drop(batch.index, inplace=True)
            df.to_csv("data.csv", index=False)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            pbar.update(len(batch))
    
    if not completed_df.empty:
        completed_df.to_csv("completed.csv", mode='a', header=False, index=False)

if __name__ == "__main__":
    main()