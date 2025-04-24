import re

import pandas as pd
import tqdm.auto

import parse_entities_v3 as parse_entities


def translate_query(
    query: str,
    schema_old: str,
    schema_new: str,
    replacements: dict,
    pattern,
    limit_pattern,
):
    """
    Выполняет преобразование SQL запроса в запрос 1С
    """
    new_query = query[:]
    limit = ""
    match = limit_pattern.search(new_query)
    if match:
        limit = match.group(1)  # Например, "LIMIT 100"
        new_query = new_query[
            : match.start()
        ]  # Удаляем часть с LIMIT из текста для замены
    # Применяем множественные замены в оставшейся части строки
    new_query = pattern.sub(lambda m: replacements[m.group(0)], new_query)
    if limit != "":
        new_query = new_query.replace("ВЫБРАТЬ ", f"ВЫБРАТЬ ПЕРВЫЕ {limit} ")

    new_query = re.sub(r"\s{2,}", " ", new_query).strip()

    mapping_by_entity, table_mapping = parse_entities.get_mapping_struct(
        schema_old, schema_new
    )
    alias_map = parse_entities.get_alias_mapping(query)
    new_query = parse_entities.replace_by_mapping(
        new_query, mapping_by_entity, alias_map
    )
    new_query = parse_entities.replace_table_names(new_query, table_mapping)

    return new_query


def run():
    # Читаем датасет
    df = pd.read_parquet("dataset/data/spider.parquet")

    # Читаем схему БД 1С
    df_schema_1c = pd.read_csv("dataset/data/schema-1c.csv", sep=";")
    df_schema_1c.columns = ["db_id", "schema_1c"]

    # Читаем схему БД Spider
    df_schema = pd.read_csv("dataset/data/schema-spider.csv", sep=";")

    # Формируем DataFrame со схемами 1С и Spider
    df_schema_common = pd.merge(
        df_schema[["db_id", "schema"]], df_schema_1c, how="inner", on="db_id"
    )

    # Добавляем в DataFrame схемы БД
    df = pd.merge(
        df[["db_id", "query", "question"]], df_schema_common, how="inner", on="db_id"
    )

    # Удаляем строки с неподдерживаемыми в 1С конструкциями
    df = df.drop(df[df["query"].str.contains("INTERSECT")].index)
    df = df.drop(df[df["query"].str.contains("EXCEPT")].index)

    # Словарь замен ключевых слов SQL на аналоги 1С
    replacements = {
        "SELECT ": "ВЫБРАТЬ ",
        " FROM ": " ИЗ ",
        " AS ": " КАК ",
        " WHERE ": " ГДЕ ",
        " BETWEEN ": " МЕЖДУ ",
        " AND ": " И ",
        " OR ": " ИЛИ ",
        " UNION ": " ОБЪЕДИНИТЬ ВСЕ ",
        " JOIN ": " ВНУТРЕННЕЕ СОЕДИНЕНИЕ ",
        " ON ": " ПО ",
        " ORDER BY ": " УПОРЯДОЧИТЬ ПО ",
        " DESC": " УБЫВ",
        " ASC": " ВОЗР",
        " GROUP BY ": " СГРУППИРОВАТЬ ПО ",
        " HAVING ": " ИМЕЮЩИЕ ",
        "max(": "МАКСИМУМ(",
        "min(": "МИНИМУМ(",
        "sum(": "СУММА(",
        "avg(": "СРЕДНЕЕ(",
        "count(": "КОЛИЧЕСТВО(",
        "COUNT(": "КОЛИЧЕСТВО(",
        "COUNT (": "КОЛИЧЕСТВО (",
        "DISTINCT ": "РАЗЛИЧНЫЕ ",
        "!=": "<>",
        " NOT IN ": " НЕ В ",
        " IN ": " В ",
        " LIKE ": " ПОДОБНО ",
        " YEAR ": " year ",
        "'": '"',
    }

    # Регулярное выражение, которое ищет любой из ключей
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))

    # Регулярное выражение для заменые LIMIT на ПЕРВЫЕ
    limit_pattern = re.compile(r"\s*LIMIT\s+(\d+)\s*$")

    tqdm.auto.tqdm.pandas()

    # Выполняем преобразование запросов
    print("Конвертация SQL-запросов в запросы 1С")
    df["query_ru"] = df.progress_apply(
        lambda x: translate_query(
            x["query"],
            x["schema"],
            x["schema_1c"],
            replacements,
            pattern,
            limit_pattern,
        ),
        axis=1,
    )

    # Сохраняем результат в файлы
    df[["query", "query_ru"]].to_csv("dataset/data/queries-ru.csv", sep=";")
    df.to_parquet("dataset/data/spider.parquet")


if __name__ == "__main__":
    run()
