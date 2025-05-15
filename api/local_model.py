import re

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate(messages, tokenizer, model, schema_proc):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        min_p=0,
        logits_processor=LogitsProcessorList([schema_proc]),
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content = tokenizer.decode(
    #     output_ids[:index], skip_special_tokens=True
    # ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content


def prepare_sequences(tokenizer):
    table_keyword_sequences = [
        tuple(tokenizer(" ИЗ", add_special_tokens=False).input_ids),
        tuple(tokenizer(" СОЕДИНЕНИЕ", add_special_tokens=False).input_ids),
    ]
    dot_sequence = tuple(tokenizer(".", add_special_tokens=False).input_ids)
    normal_keyword_sequences = [
        tuple(tokenizer(" СГРУППИРОВАТЬ", add_special_tokens=False).input_ids),
        tuple(tokenizer(" ИМЕЮЩИЕ", add_special_tokens=False).input_ids),
        tuple(tokenizer(" ВНУТРЕННЕЕ", add_special_tokens=False).input_ids),
        tuple(tokenizer(" ЛЕВОЕ", add_special_tokens=False).input_ids),
    ]

    return table_keyword_sequences, dot_sequence, normal_keyword_sequences


def find_last_keyword_position(sequence, keyword_sequences):
    max_pos = -1
    state = None
    for kw_seq in keyword_sequences:
        kw_len = len(kw_seq)
        for i in range(len(sequence) - kw_len + 1):
            if sequence[i : i + kw_len] == list(kw_seq):
                end_pos = i + kw_len - 1
                if end_pos > max_pos:
                    max_pos = end_pos
                    state = "table"
    return max_pos, state


def find_last_dot_position(sequence, dot_sequence):
    dot_seq = list(dot_sequence)
    dot_len = len(dot_seq)
    for i in range(len(sequence) - dot_len + 1):
        if sequence[i : i + dot_len] == dot_seq:
            return i + dot_len - 1
    return -1


class SchemaLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        table_sequences,
        column_sequences,
        table_keyword_sequences,
        dot_sequence,
        normal_keyword_sequences,
    ):
        self.tokenizer = tokenizer
        self.table_sequences = table_sequences  # Последовательности токенов для таблиц
        self.column_sequences = (
            column_sequences  # Последовательности токенов для колонок
        )
        self.table_keyword_sequences = table_keyword_sequences  # "ИЗ" и "СОЕДИНЕНИЕ"
        self.dot_sequence = dot_sequence  # Точка "."
        self.normal_keyword_sequences = (
            normal_keyword_sequences  # "СГРУППИРОВАТЬ", "ИМЕЮЩИЕ", и т.д.
        )
        self.in_table_name = False  # Флаг для отслеживания генерации названия таблицы

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sequence = input_ids[0].tolist()  # Берем первую последовательность из батча

        # Проверяем, есть ли ключевое слово для перехода в "normal"
        for kw_seq in self.normal_keyword_sequences:
            kw_len = len(kw_seq)
            if len(sequence) >= kw_len and sequence[-kw_len:] == list(kw_seq):
                self.in_table_name = False
                state = "normal"
                prefix = []
                break
        else:
            # Ищем последнее ключевое слово "ИЗ" или "СОЕДИНЕНИЕ"
            last_kw_pos, state = find_last_keyword_position(
                sequence, self.table_keyword_sequences
            )
            if state == "table":
                self.in_table_name = True
                prefix = sequence[last_kw_pos + 1 :]  # Токены после ключевого слова
            elif self.in_table_name:
                # Если мы в процессе генерации таблицы, проверяем её завершение
                if self.is_end_of_table_name(sequence):
                    self.in_table_name = False
                    state = "normal"
                    prefix = []
                else:
                    state = "table"
                    prefix = (
                        sequence[last_kw_pos + 1 :] if last_kw_pos != -1 else sequence
                    )
            else:
                # Проверяем точку для перехода в "column"
                dot_pos = find_last_dot_position(sequence, self.dot_sequence)
                if dot_pos != -1:
                    state = "column"
                    prefix = sequence[dot_pos + 1 :]
                else:
                    state = "normal"
                    prefix = []

        # Ограничиваем токены в зависимости от состояния
        if state == "table" and self.in_table_name:
            possible_next_tokens = set()
            for table_seq in self.table_sequences:
                if prefix == list(table_seq[: len(prefix)]):
                    if len(table_seq) > len(prefix):
                        possible_next_tokens.add(table_seq[len(prefix)])
            if possible_next_tokens:
                mask = torch.full_like(scores, -float("inf"))
                mask[:, list(possible_next_tokens)] = 0
                return scores + mask
            return scores

        elif state == "column":
            possible_next_tokens = set()
            for column_seq in self.column_sequences:
                if prefix == list(column_seq[: len(prefix)]):
                    if len(column_seq) > len(prefix):
                        possible_next_tokens.add(column_seq[len(prefix)])
            if possible_next_tokens:
                mask = torch.full_like(scores, -float("inf"))
                mask[:, list(possible_next_tokens)] = 0
                return scores + mask
            return scores

        return scores  # В состоянии "normal" не ограничиваем токены

    def is_end_of_table_name(self, sequence):
        # Проверяем, закончилось ли название таблицы (например, пробел или "КАК")
        last_token = sequence[-1]
        space_token = self.tokenizer(" ", add_special_tokens=False).input_ids[0]
        as_token = self.tokenizer("КАК", add_special_tokens=False).input_ids[0]
        return last_token == space_token or last_token == as_token


def load_model():
    peft_model_id = "checkpoints/qwen3-1_7b-lora-test-sft"

    model = AutoModelForCausalLM.from_pretrained(
        peft_model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    return model, tokenizer


def predict(
        messages,
        schema,
        model,
        tokenizer,
        table_keyword_sequences,
        dot_sequence,
        normal_keyword_sequences) -> str:
    
    schema_struct = parse_entities(schema)

    # Получаем последовательности токенов для таблиц
    table_token_sequences = [
        tuple(tokenizer(f" {table}", add_special_tokens=False).input_ids)
        for table in schema_struct.keys()
    ]

    # Получаем последовательности токенов для полей
    column_token_sequences = [
        tuple(tokenizer(column, add_special_tokens=False).input_ids)
        for columns in schema_struct.values()
        for column in columns.keys()
    ]

    # Инициализация процессора
    schema_proc = SchemaLogitsProcessor(
        tokenizer,
        table_token_sequences,  # Последовательности токенов названий таблиц
        column_token_sequences,  # Последовательности токенов названий колонок
        table_keyword_sequences,
        dot_sequence,
        normal_keyword_sequences
    )

    try:
        result = generate(messages, tokenizer, model, schema_proc)
    except Exception as e:
        print(e)
        result = ""

    return result
