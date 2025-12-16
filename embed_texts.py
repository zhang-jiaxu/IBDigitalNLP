#!/usr/bin/env python3
"""
Embed texts using SentenceTransformers on Snellius GPU.

Automatically appends the model name to the output file to prevent overwriting.
"""

import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings using SentenceTransformers.")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing text data."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings",   # now a *prefix*, not a full filename
        help="Output filename prefix (model name will be appended automatically)."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="specify Hugging Face model ID."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    # check if output directory exists
    output_dir = "embeddings"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=== Loading data ===")
    df = pd.read_csv(args.input)

    print(f"=== Loading model: {args.model} ===")
    model = SentenceTransformer(args.model)

    # get the list of answer columns
    answer_cols = df.iloc[:, 1:].columns  # exclude ID column

    for col in answer_cols:
        if col == "Wish_predictability_flare":
            continue  # skip this column as it's used for differentiation only
        
        print(f"Preparing texts from column: {col}")
        
        # select the ID and current column
        df_col = df[["ID", col]].copy()

        if col == "Wish_predictability_because":
            df_col = df[["ID", "Wish_predictability_flare", col]].copy()
        
        # mask: entries with at least one space (multi-word answers)
        mask = df_col[col].fillna("").str.strip().str.contains(" ")
        
        # filtered text for embedding
        df_docs_for_embedding = df_col.loc[mask].copy()
            
        # list of texts for embedding
        texts = df_docs_for_embedding[col].astype(str).tolist()

        # save ONLY texts used for embedding
        df_docs_for_embedding.to_csv(
            f"data/answers_for_embedding_{col}.csv", 
            index=False
        )
    
        print(f"Embedding texts from column: {col}, length: {len(texts)}")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        output_file = os.path.join(output_dir, f"{args.output}_{col}.npy")
        print(f"Embeddings shape for {col}: {embeddings.shape}")
        print(f"=== Saving embeddings to {output_file} ===")
        np.save(output_file, embeddings)    

    print("Done.")


if __name__ == "__main__":
    main()
