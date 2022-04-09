import json
from pathlib import Path
import os
import shutil
import pandas as pd
from utils.utils import clean_text
from tqdm import tqdm

with open("./config.json", "r") as conf:
    config = json.loads(conf.read())


def create_csvs(data_json, csv_path):
    csv = pd.DataFrame()
    for id in tqdm(range(len(data_json["data"])), total=len(data_json["data"])):
        paragraphs = data_json["data"][id]["paragraphs"]
        for para in paragraphs:
            qas = para["qas"]
            context = para["context"]
            context = clean_text(context)
            for q in qas:
                question = clean_text(q["question"])
                if len(question) < config.get("question_length_threshold"):
                    continue
                if q["is_impossible"]:
                    continue
                for answer in q["answers"]:
                    answer_text = clean_text(answer["text"])
                    answer_start = answer["answer_start"]
                val = {
                    "question": question,
                    "context": context,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                }
                csv = csv.append(val, ignore_index=True)
    csv.to_csv(csv_path, index=False)


if __name__ == "__main__":
    dataset_path = Path(config.get("qa_path"))
    if dataset_path.is_dir():
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)
    for key_path, type in [("train_path", "train"), ("val_path", "val")]:
        print(f"-----Creating {type} csv -----------")
        with open(config.get(key_path), "r") as data_path:
            data_json = json.load(data_path)
        csv_path = dataset_path / Path(type + ".csv")
        print(csv_path.as_posix())
        create_csvs(data_json, csv_path)
        print(f"------------Created {type} csv in {csv_path}--------------------")
