import pandas as pd


def run():
    # Читаем датасет
    df = pd.read_parquet("dataset/data/spider.parquet")

    # Читаем DataFrame с запросами, которые были успешно выполнены в тестовой базе 1С
    df_executed = pd.read_csv("dataset/data/queries-ru-executed.csv", sep=";", index_col=0)

    # Создаем DataFrame, в котором находятся только корректные запросы 1С
    df = df[["question", "schema_1c", "query_ru"]].loc[df_executed.index]

    # Сохраняем вопросы пользователей на английском языке
    df["question"].to_csv("dataset/data/questions.csv", sep=";")
    
    df.to_parquet("dataset/data/spider.parquet")


if __name__ == "__main__":
    run()
