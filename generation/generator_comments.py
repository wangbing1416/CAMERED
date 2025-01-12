import os

import torch
import json
import argparse
import pandas as pd
from mainMoE import T5WithLoRALayer



def parse_args():
    parser = argparse.ArgumentParser(description="Generate comments for tweets using a model with LoRA layers.")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_tsv_path", type=str, required=True, help="Path to the output TSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--linear_layer_path", type=str, required=True, help="Path to the linear layer.")
    parser.add_argument("--num_experts", type=int, default=10, help="Number of experts for LoRA.")
    parser.add_argument("--K", type=int, default=16, help="Value of K for LoRA.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for LoRA.")
    return parser.parse_args()


def load_model(args, device):
    print(f"Loading model and tokenizer from '{args.model_path}'")
    try:
        model = T5WithLoRALayer.load_linear_layer(
            model_name_or_path=args.model_path,
            linear_layer_path=args.linear_layer_path,
            device=device,
            num_experts=args.num_experts,
            K=args.K,
            threshold=args.threshold
        )
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit()


def generate_comments(model, input_text, device):
    try:
        inputs = model.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
    except Exception as e:
        print(f"Error encoding input text: {e}")
        return []

    generation_args = {
        "temperature": 1.5,
        "top_k": 30,
        "top_p": 0.8,
        "do_sample": True,
        "max_length": 60,
        "num_return_sequences": 10
    }

    threads = 0.5
    try:
        output_ids = model.generateself(
            input_ids=input_ids,
            attention_mask=attention_mask,
            threads=threads,
            **generation_args
        )
        comments = [model.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for output_id in output_ids]
        return comments
    except Exception as e:
        print(f"Error generating comments: {e}")
        return []


def process_json(input_json_path, model, device, output_tsv_path):
    with open(input_json_path, 'r', encoding='utf-8') as infile:
        data = []
        for line in infile:
            item = json.loads(line)
            tweet_id = item.get('id', '')  # Extract tweet id
            tweet = item.get('tweet', item.get('content', ''))

            # Generate comments for the tweet
            new_comments = generate_comments(model, tweet, device)
            if not new_comments:
                new_comments = []  # If generation fails, use empty list

            # Join the comments with '||' separator
            comments_str = ' || '.join(new_comments)

            # Store the result as a tuple of tweet id and comments
            data.append((tweet_id, comments_str))

        # Write results to TSV file
        df = pd.DataFrame(data, columns=['id', 'comments'])
        df.to_csv(output_tsv_path, sep='\t', index=False)


if __name__ == "__main__":
    args = parse_args()

    # Set device (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args, device)

    # Process JSON and write results to TSV
    process_json(args.input_json_path, model, device, args.output_tsv_path)
    print(f"Comments generated and saved to '{args.output_tsv_path}'")
