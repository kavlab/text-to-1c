import os

import pandas as pd

GOOGLE_TRANSLATOR = "google"
DEEPL_TRANSLATOR = "deepl"


def translate_with_google(df: pd.DataFrame):
    import tqdm.auto
    from deep_translator import GoogleTranslator

    tqdm.auto.tqdm.pandas()

    translator = GoogleTranslator(source="en", target="ru")
    df["question_ru"] = df["question"].progress_apply(lambda x: translator.translate(x))


def translate_with_deepl(df: pd.DataFrame):
    import deepl
    import tqdm.auto
    from secret import DEEPL_API_KEY

    tqdm.auto.tqdm.pandas()

    deepl_client = deepl.DeepLClient(DEEPL_API_KEY)

    df["question_ru"] = df["question"].progress_apply(
        lambda x: deepl_client.translate_text(x, source_lang="EN", target_lang="RU")
    )


def run(translator_type: str = None):
    # Загружаем основной датасет
    df = pd.read_parquet("dataset/data/spider.parquet")

    # Файл для кэша переводов
    questions_file = f"dataset/data/questions-ru-{translator_type}.csv"

    # Проверяем, есть ли уже файл с переводами
    if translator_type in (GOOGLE_TRANSLATOR, DEEPL_TRANSLATOR) and os.path.exists(
        questions_file
    ):
        print(f"Загружаем готовые переводы из {questions_file}")
        cache_df = pd.read_csv(questions_file, sep=";")
        # Объединяем столбец question_ru из кеша по индексу
        df = df.join(cache_df[['question_ru']], how='left')
    else:
        # Если файл не найден или не указан тип переводчика
        if translator_type == GOOGLE_TRANSLATOR:
            print("Перевод вопросов пользователей с помощью Google Translate")
            translate_with_google(df)
        elif translator_type == DEEPL_TRANSLATOR:
            print("Перевод вопросов пользователей с помощью DeepL")
            translate_with_deepl(df)
        else:
            print("Не указан тип используемого переводчика")
            return

        # Сохраняем кэш переводов для последующей загрузки
        df[["question", "question_ru"]].to_csv(questions_file, sep=";", index=False)

    # Сохраняем обновленный датасет
    df.to_parquet("dataset/data/spider.parquet")


if __name__ == "__main__":
    run(DEEPL_TRANSLATOR)
