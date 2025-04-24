import pandas as pd

GOOGLE_TRANSLATOR = "google"
DEEPL_TRANSLATOR = "deepl"


def translate_with_google(df: pd.DataFrame):
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="en", target="ru")
    df["question_ru"] = df["question"].progress_apply(lambda x: translator.translate(x))


def translate_with_deepl(df: pd.DataFrame):
    import deepl
    from secret import DEEPL_API_KEY

    deepl_client = deepl.DeepLClient(DEEPL_API_KEY)

    df["question_ru"] = df["question"].progress_apply(
        lambda x: deepl_client.translate_text(x, source_lang="EN", target_lang="RU")
    )


def run(translator_type: str = None):
    df = pd.read_parquet("dataset/data/spider.parquet")

    if translator_type == GOOGLE_TRANSLATOR:
        print("Перевод вопросов пользователей")
        translate_with_google(df)
    elif translator_type == DEEPL_TRANSLATOR:
        print("Перевод вопросов пользователей")
        translate_with_deepl(df)
    else:
        print("Не указан тип используемого переводчика")
        return
    
    # Сохраняем вопросы на английском и русском языках
    df[["question", "question_ru"]].to_csv("dataset/data/questions-ru.csv", sep=";")

    df.to_parquet("dataset/data/spider.parquet")


if __name__ == "__main__":
    run()
