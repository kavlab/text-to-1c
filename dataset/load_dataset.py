import pandas as pd
from datasets import load_dataset


def run():
    # Загружаем датасет Spider 1.0 с HuggingFace
    dataset = load_dataset("xlangai/spider")

    # Загружаем схемы БД с HuggingFace, используемые в Spider
    dataset_schema = load_dataset("richardr1126/spider-schema")

    # Конвертируем train и validation выборки в DataFrame
    df_train = dataset["train"].to_pandas()
    df_validation = dataset["validation"].to_pandas()

    # Объединяем train и validation в один DataFrame, т.к.
    # нужно будет подготовить общий датасет для Text-to-1C
    df = pd.concat([df_train, df_validation], ignore_index=True)

    # Сохраняем полученный датасет для обработки в других скриптах
    df.to_parquet("dataset/data/spider.parquet")

    # Конвертируем датасет со схемой в DataFrame (train содержит все схемы)
    df_schema = dataset_schema["train"].to_pandas()
    df_schema.columns = ["db_id", "schema", "primary_keys", "foreing_keys"]

    # Сохраняем схему в CSV для последующей конвертации в схему БД 1С
    df_schema[["db_id", "schema"]].to_csv("dataset/data/schema-spider.csv", sep=";")


if __name__ == "__main__":
    run()
