import json

import pandas as pd
from sklearn.model_selection import train_test_split


def save_dataframe_to_json(df: pd.DataFrame, filename: str):
    """
    Сохраняет DataFrame в файлы в формате JSON
    """
    with open(filename, "w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            json_obj = {
                "messages": [
                    {"content": row["system"], "role": "system"},
                    {"content": row["question_ru"], "role": "user"},
                    {"content": row["query_ru"], "role": "assistant"},
                ]
            }
            file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def run():
    # Читаем датасет
    df = pd.read_parquet("dataset/data/spider.parquet")

    # Добавляем системного сообщение для LLM
    df["system"] = (
        "You are an text to SQL query translator. "
        "Users will ask you questions in Russian and "
        "you will generate a SQL query based on the provided SCHEMA.\n"
        "SCHEMA: " + df["schema_1c"]
    )

    # Из полученного DataFrame выделяем тренировочную и тестовую выборки
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=82)

    # Сохраняем выборке в формате JSON
    save_dataframe_to_json(train_df, "dataset/data/ru_train.json")
    save_dataframe_to_json(test_df, "dataset/data/ru_test.json")


if __name__ == "__main__":
    run()
