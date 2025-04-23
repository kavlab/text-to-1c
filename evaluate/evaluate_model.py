import re

# Словари синонимов и соответствий языков
KEYWORD_MAP = {
    # Русские ключевые слова -> Английские эквиваленты
    "ВЫБРАТЬ": "SELECT",
    "ИЗ": "FROM",
    "ГДЕ": "WHERE",
    "СГРУППИРОВАТЬ ПО": "GROUP BY",
    "ИМЕЮЩИЕ": "HAVING",
    "УПОРЯДОЧИТЬ ПО": "ORDER BY",
    "КАК": "AS",
}
LOGICAL_MAP = {
    # Логические операторы и константы
    "И": "AND",
    "ИЛИ": "OR",
    "НЕ": "NOT",
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "ИСТИНА": "TRUE",
    "ЛОЖЬ": "FALSE",
}


def normalize_query(query: str) -> str:
    """Приводит запрос к упрощенному каноническому виду:
    убрать лишние пробелы, заменить синонимы, привести к верхнему регистру."""
    # Убираем начальные/конечные пробелы и лишние пробелы внутри
    text = query.strip()
    # Заменяем запятые и скобки пробелами вокруг, чтобы они учитывались
    # как отдельные токены при split, если нужно
    text = re.sub(r"([\(\),])", r" \1 ", text)
    # Приводим к верхнему регистру
    text = text.upper()
    # Заменяем ключевые слова (более длинные сначала, чтобы не перепутать часть фразы)
    # Например, "СГРУППИРОВАТЬ ПО" должен замениться целиком, 
    # а не по отдельности "СГРУППИРОВАТЬ" или "ПО".
    for rus, eng in sorted(KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(rus, eng)
    # Заменяем логические операторы/константы
    for syn, std in LOGICAL_MAP.items():
        # добавляем пробелы вокруг, чтобы заменить только как отдельные слова
        text = re.sub(rf"\b{syn}\b", std, text)
    # Убираем повторные пробелы
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_query_components(query: str) -> dict:
    """Разбивает запрос 1С на компоненты. Предполагается один SELECT без вложенных запросов."""
    normalized = normalize_query(query)
    components = {
        "SELECT": "",
        "FROM": "",
        "WHERE": "",
        "GROUP BY": "",
        "HAVING": "",
        "ORDER BY": "",
    }
    # Найдём позиции ключевых секций в строке
    # Для надёжности ищем с пробелом или концом строки,
    # чтобы не перепутать, например, "WHERE" внутри имени поля.
    indices = {}
    for key in components.keys():
        # Регекс для ключевого слова как отдельного (например, r'\bSELECT\b')
        match = re.search(rf"\b{key}\b", normalized)
        if match:
            indices[key] = match.start()
    if "SELECT" not in indices:
        # Если нет SELECT, возвращаем пустые компоненты
        return components

    # Определяем порядок секций, которые присутствуют, по их стартовым индексам
    present_sections = sorted(indices.items(), key=lambda x: x[1])
    # Добавляем фиктивный конец строки как границу последней секции
    present_sections.append(("END", len(normalized)))

    # Выделяем текст между текущей секцией и началом следующей
    for i in range(len(present_sections) - 1):
        sec_name, start_idx = present_sections[i]
        _, next_start = present_sections[i + 1]
        if sec_name == "SELECT":
            # Отрезаем "SELECT" + пробел, чтобы оставить только содержимое
            comp_text = normalized[start_idx + len("SELECT") : next_start].strip()
        else:
            comp_text = normalized[start_idx + len(sec_name) : next_start].strip()
        components[sec_name] = comp_text

    return components


def component_matching_f1(pred_query: str, ref_query: str) -> dict:
    """Рассчитывает F1-score для каждой компоненты между
    запросом модели и референсным запросом."""
    pred_comp = parse_query_components(pred_query)
    ref_comp = parse_query_components(ref_query)
    scores = {}
    for comp in ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY"]:
        pred_text = pred_comp.get(comp, "")
        ref_text = ref_comp.get(comp, "")
        # Разбиваем на токены (простое разбиение по пробелам и запятым, которые уже окружены пробелами)
        pred_tokens = set(
            token for token in pred_text.split() if token not in {"", ","}
        )
        ref_tokens = set(token for token in ref_text.split() if token not in {"", ","})
        if len(ref_tokens) == 0 and len(pred_tokens) == 0:
            # Если ни в рефе, ни в предсказании компоненты нет, считаем идеальным (можно пропустить)
            f1 = 1.0
        elif len(ref_tokens) == 0:
            # Если в эталоне компоненты нет, а в предсказании есть лишняя компонента
            f1 = 0.0
        elif len(pred_tokens) == 0:
            # В предсказании отсутствует обязательная компонента
            f1 = 0.0
        else:
            # Вычисляем Precision, Recall, F1
            true_positives = len(pred_tokens & ref_tokens)
            precision = true_positives / len(pred_tokens)
            recall = true_positives / len(ref_tokens)
            f1 = (
                0.0
                if (precision + recall) == 0
                else 2 * precision * recall / (precision + recall)
            )
        scores[comp] = round(f1, 4)
    return scores


def batch_component_matching_f1(predicted_queries, reference_queries):
    """
    Рассчитывает средние значения F1-score для каждой компоненты
    (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY)
    по списку пар предсказанный/эталонный запрос.

    :param predicted_queries: список сгенерированных (предсказанных) запросов (строки)
    :param reference_queries: список эталонных (референсных) запросов (строки)
    :return: словарь, где ключи – компоненты запроса, значения – средний F1-score по всем парам.
    """
    if len(predicted_queries) != len(reference_queries):
        raise ValueError(
            "Количество предсказанных и эталонных запросов должно совпадать."
        )

    # Инициализируем суммарные значения F1 для каждой компоненты
    components = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY"]
    cumulative_scores = {comp: 0.0 for comp in components}

    num_examples = len(predicted_queries)

    for pred, ref in zip(predicted_queries, reference_queries):
        scores = component_matching_f1(pred, ref)
        for comp in components:
            cumulative_scores[comp] += scores.get(comp, 0.0)

    # Вычисляем среднее значение для каждой компоненты
    average_scores = {
        comp: round(cumulative_scores[comp] / num_examples, 4) for comp in components
    }
    return average_scores
