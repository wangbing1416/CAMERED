import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import json
import re
import spacy
import emoji
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


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


def postprocess(text):
    return text.replace(".", "").replace('</>', '')


def generate_description(keyword, tokenizer, model, device):
    input_text = f"{keyword} is defined as"
    encoding = tokenizer(text=[input_text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(
        device)

    output = model.generate(
        **encoding,
        return_dict_in_generate=True,
        output_scores=False,
        max_length=512,
        temperature=0.5,
        do_sample=True,
        repetition_penalty=3.0,
        top_k=50
    )

    description = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0]
    description = postprocess(description)

    return description


def main():
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        contents_data = f.readlines()

    nlp = spacy.load('zh_core_web_trf')

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_model)

    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)

    results = []
    contents_id = 0

    for contents_json in contents_data:
        content = json.loads(contents_json)
        content_text = content.get('content', '')

        if content_text:
            doc = nlp(content_text)
            entities = [ent.text for ent in doc.ents]
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]

            unique_keywords = list(set(entities + nouns))

            for keyword in unique_keywords:
                description = generate_description(keyword, tokenizer, model, device)
                description = clean_tweet(replace_emojis_with_text(unicode_to_emoji(description)))
                results.append([content_text, description])

        contents_id += 1
        print(contents_id)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"results have saved in {args.output_file}")


if __name__ == "__main__":
    main()