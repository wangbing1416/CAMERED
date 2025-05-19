# CAMERED
This repo is the released code of our **SIGIR 2025** work **Collaboration and Controversy Among Experts: Rumor Early Detection by Tuning a Comment Generator** and our extended paper **Multi-View Controversy Among Multi-Expert Comments for Rumor Early Detection**.


### 0 - Requirements

```
torch==2.1.0
cudatoolkit==12.5
transformers==4.36.2
```

### 1 - Train and Generate Comments

1. Prepare data from their original papers, and put them in `./data/`.
2. Process _Twitter_ and _Weibo_ data, e.g.,
```shell
cd generation
python data_twitter_preprocess.py --input_folder "../data/twitter15/" --output_folder "../data/twitter15/"
python data_weibo_preprocess.py --input_folder "../data/weibo16/" --output_folder "../data/weibo16/"
```
3. Extract entities from data, and obtain their entity descriptions, e.g.,
```shell
python twitter_entity_extraction.py --input_file "../data/twitter15/train.json" --output_file "../data/twitter15/entity_description.json" --llm_model "../gpt-2/"
python weibo_entity_extraction.py --input_file "../data/weibo16/train.json" --output_file "../data/weibo16/entity_description.json" --llm_model "../chinese-t5-base/"
```
4. Train the generator, e.g.,
```shell
python train_generator.py
```
5. Generate comments, e.g.,
```shell
python generator_comments.py --input_json_path "../data/twitter16/test.json" --output_tsv_path "../data/twitter16/generated_comments.tsv" --model_path "../flan-t5-base/" --linear_layer_path "../data/twitter16/gan/final_generator/" --num_experts 10 --K 16 --threshold 0.5
```

### 2 - Train Detection Models

1. Prepare generated data, put the generated comments in `./data/[DATASET]/generated`, and rename them as `generated_train.json` (we provide a test case in the folder)
2. Train and test detection models, e.g.,
```shell
cd detection
python main.py --dataset [DATASET] --model [MODEL]
```
DATASET in `twitter15`, `twitter16`, `weibo16`, `weibo20`, and MODEL in `cbert`, `defend`, `casfend`, `kahan`, `cameredit`

3. The evaluation metrics are saved in `./detection/json/[DATASET]`, and you can run `./read_json.py` to automatically read the json files into an excel table


### Citation
```

```
