import re
import json
import argparse
from pathlib import Path
import emoji


def unicode_to_emoji(unicode_text):
    try:
        return unicode_text.encode('utf-16', 'surrogatepass').decode('utf-16')
    except UnicodeEncodeError:
        return unicode_text  # if decode fail, return original text


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


def detect_json_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return 'array'
        elif first_char == '{':
            return 'object'
        else:
            return 'unknown'


label_counter = 1


def preprocess_json(input_dir, output_dir):
    global label_counter
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in input_dir.glob("*.json"):
        output_file = output_dir / json_file.name
        json_format = detect_json_format(json_file)
        print(f"Processing file: {json_file} as format: {json_format}")

        if json_format == 'array':
            with open(json_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                try:
                    data = json.load(infile)
                    if not isinstance(data, list):
                        print(f"Unexpected JSON array structure in file {json_file}")
                        continue
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON array in file {json_file}: {e}")
                    continue

                for entry_num, entry in enumerate(data, 1):
                    if not isinstance(entry, dict):
                        print(f"Warning: Entry {entry_num} in file {json_file} is not a dict.")
                        continue

                    label = entry.get('label')
                    content = entry.get('content')
                    comments = entry.get('comments', [])
                    id = label_counter
                    label_counter += 1
                    # map labels
                    if label == "real":
                        label = 1
                    elif label == "fake":
                        label = 0

                    if not content:
                        content = ''
                    else:
                        content = clean_tweet(content)

                    cleaned_comments = [clean_tweet(comment) for comment in comments]

                    new_entry = {
                        "id": id,
                        "label": label,
                        "content": content,
                        "comments": cleaned_comments
                    }

                    json.dump(new_entry, outfile, ensure_ascii=False)
                    outfile.write('\n')

        elif json_format == 'object':
            with open(json_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if not isinstance(entry, dict):
                            print(f"Warning: Entry at line {line_num} in file {json_file} is not a dict.")
                            continue
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {json_file} at line {line_num}: {e}")
                        continue

                    label = entry.get('label')
                    content = entry.get('content')
                    comments = entry.get('comments', [])
                    id = label_counter
                    label_counter += 1
                    if label == "real":
                        label = 1
                    elif label == "fake":
                        label = 0

                    if not content:
                        content = ''
                    else:
                        content = clean_tweet(content)

                    cleaned_comments = [clean_tweet(comment) for comment in comments]

                    new_entry = {
                        "id": id,
                        "label": label,
                        "content": content,
                        "comments": cleaned_comments
                    }

                    json.dump(new_entry, outfile, ensure_ascii=False)
                    outfile.write('\n')

        else:
            print(f"Unsupported JSON format in file {json_file}")
            continue


def main():
    parser = argparse.ArgumentParser(description="preprocess json file")
    parser.add_argument('--input_folder', type=str, required=True, help='input json file')
    parser.add_argument('--output_folder', type=str, required=True, help='output json file')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    preprocess_json(input_folder, output_folder)


if __name__ == "__main__":
    main()
