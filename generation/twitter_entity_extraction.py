import json
import spacy
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import re
import emoji


def unicode_to_emoji(unicode_text):
    try:
        return unicode_text.encode('utf-16', 'surrogatepass').decode('utf-16')
    except UnicodeEncodeError:
        return unicode_text


def replace_emojis_with_text(text):
    return emoji.demojize(text)


def clean_tweet(tweet):
    # replace urls
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]{2}))+", " ", tweet)
    # replace @
    text = re.sub(r"@\w+", " ", text)
    # replace \n
    text = text.replace("\n", " ")
    # replace special tokens
    replacements = {
        "'s": "",
        "'": "",
        "n’t": " n’t",
        "#": "",
        "_": " ",
        "-": " ",
        "&amp;": "",
        "&gt;": "",
        "\"": "",
        ".": "",
        ",": "",
        "(": "",
        ")": "",
        ":": "",
        ";": "",
        "!": "",
        "?": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)
    # remove spaces
    text = re.sub(r'\s+', ' ', text)
    # turn lower
    text = text.lower()
    text = text.strip()
    # replace emojis
    text = replace_emojis_with_text(unicode_to_emoji(text))

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions for tweets.")
    parser.add_argument('--input_file', type=str, required=True, help='input json file path')
    parser.add_argument('--output_file', type=str, required=True, help='output json file path')
    parser.add_argument('--llm_model', type=str, required=True, help='llm path')
    return parser.parse_args()


def generate_description(keyword, tokenizer, model, device):
    input_text = f"{keyword} is defined as"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    attention_mask = torch.ones(input_ids.shape, device=device)
    pad_token_id = tokenizer.eos_token_id

    # generate descriptions
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=50,
        min_length=20,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_beams=3,
        early_stopping=True
    )

    description = tokenizer.decode(output[0], skip_special_tokens=True)

    if '.' in description:
        description = description[:description.rfind('.') + 1]

    return description


def main():
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        tweets_data = f.readlines()

    # load SpaCy model
    nlp = spacy.load('zh_core_web_trf')

    # load llm and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.llm_model)
    model = GPT2LMHeadModel.from_pretrained(args.llm_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    tweet_id = 0

    for tweet_json in tweets_data:
        tweet = json.loads(tweet_json)
        tweet_text = tweet.get('tweet', '')

        if tweet_text:
            # extract entities
            doc = nlp(tweet_text)
            entities = [ent.text for ent in doc.ents]
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]  # 提取名词

            # combine entities
            unique_keywords = list(set(entities + nouns))

            for keyword in unique_keywords:
                description = generate_description(keyword, tokenizer, model, device)
                description = clean_tweet(replace_emojis_with_text(unicode_to_emoji(description)))
                results.append([tweet_text, description])

        tweet_id += 1

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"results have saved in {args.output_file}")


if __name__ == "__main__":
    main()

