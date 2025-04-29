import convert_queries
import create_final_dataset
import filter_exec_queries
import load_dataset
import translate_questions

# Загрузка датасета Spider с HuggingFace и подготовка для дальнейшей обработки
load_dataset.run()

# Конвертация SQL-запросов в запросы на языке 1С
convert_queries.run()

# Фильтрация датасета - в результате остаются только те запросы,
# которые были успешно выполнены в тестовой базе 1С
filter_exec_queries.run()

# Перевод вопросов пользователей с английского языка на русский
translate_questions.run(translate_questions.DEEPL_TRANSLATOR)

# Формирования финального датасета, который будет использоваться для обучения моделей
create_final_dataset.run()
