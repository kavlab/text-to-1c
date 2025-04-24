# Text-to-1C

[dataset](dataset) - содержит скрипты адаптации датасета Spider для задачи Text-to-1C; основной скрипт, который запускает остальные - [prepare_dataset.py](dataset/prepare_dataset.py)

[test_base](test_base) - [скрипт](test_base/create_config.py) и шаблоны для создания тестовой базы 1С, в которой проверяется выполнение запросов

[train](train) - скрипты для обучения и тестирования моделей

[evaluate](evaluate) - вычисление точечных и интервальных оценок метрик Exact match, Component match, Execution accuracy
