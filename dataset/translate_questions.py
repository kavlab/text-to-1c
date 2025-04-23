import pandas as pd
from datasets import Dataset


def translate_with_google(df: pd.DataFrame):
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source='en', target='ru')
    df['question_ru'] = df['question'].progress_apply(lambda x: translator.translate(x))

def main():
    df = pd.read_parquet("dataset/spider.parquet", sep=";")
    dataset_with_schema = Dataset.from_pandas(df)


if __name__ == "__main__":
    main()
