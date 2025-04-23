import re


def parse_entities(entity_str):
    """
    Преобразует строку в структуру вида:
    {
        'entity': {
            'field_name': 'type'
        }
    }
    Все имена приводятся к нижнему регистру для удобства сопоставления
    """
    entities = {}
    parts = [p.strip() for p in entity_str.strip().split("|")]
    for part in parts:
        if not part:
            continue
        entity_match = re.match(r"(.+?)\s*:\s*(.+)", part)
        if not entity_match:
            continue
        entity_name, fields_str = entity_match.groups()
        fields = [f.strip() for f in fields_str.split(",")]
        field_dict = {}
        for f in fields:
            field_match = re.match(r"(.+?)\s*\((.+?)\)", f)
            if field_match:
                field_name, field_type = field_match.groups()
                field_dict[field_name.strip().lower()] = field_type.strip()
        # Сохраняем имя сущности в нижнем регистре
        entities[entity_name.strip().lower()] = field_dict
    return entities


def get_mapping_struct(schema_spider: str, schema_1c: str):
    """
    Возвращает два словаря:
    1. mapping_by_entity: соответствие полей из схемы Spider и 1С.
       Формат: { 'spider_entity': { 'field_name': '1c_поле', ... } }
    2. table_mapping: соответствие имени таблицы из Spider и 1С.
       Формат: { 'spider_entity': '1c_таблица', ... }

    Порядок сопоставления берется по позициям, поэтому схемы должны быть согласованы
    """
    entities_spider = parse_entities(schema_spider)
    entities_1c = parse_entities(schema_1c)

    mapping_by_entity = {}
    table_mapping = {}

    # Используем zip, так как порядок таблиц совпадает в обеих схемах.
    for (entity_spider, fields_spider), (entity_1c, fields_1c) in zip(
        entities_spider.items(), entities_1c.items()
    ):
        field_mapping = {}
        fields_spider_order = list(fields_spider.keys())
        fields_1c_order = list(fields_1c.keys())
        for f_spider, f_1c in zip(fields_spider_order, fields_1c_order):
            field_mapping[f_spider] = f_1c  # f_spider уже в нижнем регистре
        mapping_by_entity[entity_spider] = field_mapping
        # Запоминаем соответствие имени таблицы (spider -> 1С)
        table_mapping[entity_spider] = entity_1c.strip()
    return mapping_by_entity, table_mapping


def get_alias_mapping(query: str) -> dict:
    """
    Из запроса получает словарь соответствия alias -> имя таблицы.
    Для конструкции 'FROM table' без явного alias, alias = table
    """
    pattern = r"(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?"
    alias_map = {}
    matches = re.findall(pattern, query, flags=re.IGNORECASE)
    for table, alias in matches:
        alias = alias if alias else table
        alias_map[alias] = table.lower()  # приводим имя таблицы к нижнему регистру
    return alias_map


def replace_by_mapping(query: str, mapping_by_entity: dict, alias_map: dict) -> str:
    """
    Выполняет замену полей в запросе вида "Alias.field" на "Alias.НовоеПоле"
    и, если запрос использует одну таблицу (без алиасов), заменяет поля по границам слов
    """
    # Замена для конструкций Alias.field
    pattern = r"(\w+)\.(\w+)"

    def replacement(match):
        alias = match.group(1)
        field = match.group(2)
        if alias in alias_map:
            entity = alias_map[alias]
            field_mapping = mapping_by_entity.get(entity, {})
            mapped_field = field_mapping.get(field.lower())
            if mapped_field:
                return f"{alias}.{mapped_field}"
        return match.group(0)

    new_query = re.sub(pattern, replacement, query)

    # Если в запросе используется только одна таблица (alias_map содержит один элемент)
    # и поля не указаны через alias, выполняется замена по границам слов.
    if len(alias_map) == 1:
        alias = next(iter(alias_map.keys()))
        entity = alias_map[alias]
        field_mapping = mapping_by_entity.get(entity, {})
        if field_mapping:
            keys_pattern = (
                r"\b(" + "|".join(re.escape(k) for k in field_mapping.keys()) + r")\b"
            )

            def repl(m):
                token = m.group(0)
                return field_mapping.get(token.lower(), token)

            new_query = re.sub(keys_pattern, repl, new_query, flags=re.IGNORECASE)
    return new_query


def replace_table_names(query: str, table_mapping: dict) -> str:
    """
    Заменяет имена таблиц в конструкциях FROM и JOIN на значения из table_mapping.
    Регулярное выражение находит конструкции вида:
      ИЗ table [КАК alias]
      СОЕДИНЕНИЕ table [КАК alias]
    и заменяет table на соответствующее из table_mapping (по сравнению в нижнем регистре).
    """
    pattern = re.compile(
        r"\b(ИЗ|СОЕДИНЕНИЕ)\s+(\w+)(\s+(?:КАК\s+)?\w+)?", re.IGNORECASE
    )

    def table_replacement(match):
        keyword = match.group(1)
        table_name = match.group(2)
        alias_part = match.group(3) if match.group(3) else ""
        # Пытаемся найти замену для имени таблицы (приводим его к нижнему регистру)
        new_table_name = table_mapping.get(table_name.lower(), table_name)
        return f"{keyword} {new_table_name}{alias_part}"

    return pattern.sub(table_replacement, query)
