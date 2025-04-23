import csv
import uuid

DIR = "test_base"
TEMPLATES = "templates"
CONFIG_VERSION = "d862a68eb9fa1f42bb56075079f31d7d00000000"
STANDARD_ATTRIBUTES = ["Код", "Наименование"]
STANDARD_DIMENSIONS = ["Период", "Регистратор"]
CACHE_ID = {}


def generate_uuid(name=None):
    """Генерирует новый UUID-строку."""
    id = str(uuid.uuid4())
    if name is not None:
        CACHE_ID[name] = id
    return id


def parse_metadata_csv():
    """
    Читает CSV-файл и возвращает словарь:
      { catalog_name: [(attr_name, attr_type), ...], ... }
    Элементы с именем "Ссылка" пропускаются.
    """
    catalogs = {}
    registers = {}
    with open(f"{DIR}/metadata.csv", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if len(row) < 2:
                continue
            # row[0] – индекс (не используется), row[1] – схема
            scheme = row[1]
            try:
                # Разбиваем по двоеточию: до двоеточия — имя справочника, после — список реквизитов
                metadata_part, attributes_part = scheme.split(":", 1)
            except ValueError:
                continue
            # Из имени справочника берём последний компонент (например, Справочник1)
            metadata_full = metadata_part.strip()

            if metadata_full.startswith("Справочник"):
                metadata_type = "catalog"
            elif metadata_full.startswith("РегистрСведений"):
                metadata_type = "register"
            else:
                metadata_type = None

            metadata_name = metadata_full.split(".")[-1]

            attributes = []
            # Разбиваем список по запятой
            for attr in attributes_part.split(","):
                attr = attr.strip()
                # Ожидаемый формат: "Имя (Тип)"
                if "(" in attr and ")" in attr:
                    attr_name = attr.split("(")[0].strip()
                    attr_type = attr.split("(")[1].replace(")", "").strip()
                    # Пропускаем элемент "Ссылка"
                    if metadata_type == "catalog" and attr_name.lower() == "ссылка":
                        continue
                    attributes.append((attr_name, attr_type))

            if metadata_type == "catalog":
                catalogs[metadata_name] = attributes
            elif metadata_type == "register":
                registers[metadata_name] = attributes

    return catalogs, registers


def create_config(catalogs, registers):
    """Создает файл Configuration.xml"""
    with open(f"{DIR}/{TEMPLATES}/Configuration.xml", "r") as f:
        template = f.readlines()

        new_metadata = ""

        for catalog_name, attributes in catalogs.items():
            new_metadata = (
                f"{new_metadata}\n			<Catalog>{catalog_name}</Catalog>"
            )

        for register_name, dimensions in registers.items():
            new_metadata = f"{new_metadata}\n			<InformationRegister>{register_name}</InformationRegister>"

        for index, line in enumerate(template):
            if "{NEW_METADATA}" in line:
                template[index] = template[index].replace(
                    "{NEW_METADATA}", new_metadata
                )

        with open(f"{DIR}/Configuration.xml", "wt") as f_result:
            f_result.writelines(template)


def create_config_dump(catalogs, registers):
    """Создает файл ConfigDumpInfo.xml"""
    with open(f"{DIR}/{TEMPLATES}/ConfigDumpInfo.xml", "r") as f:
        template = f.readlines()

        new_metadata = ""

        for catalog_name, attributes in catalogs.items():
            metadata_name = f"Catalog.{catalog_name}"
            id = generate_uuid(metadata_name)
            new_metadata = f'{new_metadata}		<Metadata name="{metadata_name}" id="{id}" configVersion="{CONFIG_VERSION}">'

            for attr_name, attr_type in attributes:
                if attr_name in STANDARD_ATTRIBUTES:
                    continue
                meta_attr_name = f"{metadata_name}.Attribute.{attr_name}"
                id = generate_uuid(meta_attr_name)
                new_metadata = f'{new_metadata}\n			<Metadata name="{meta_attr_name}" id="{id}"/>'

            new_metadata = f"{new_metadata}\n		</Metadata>\n"

        for register_name, dimensions in registers.items():
            metadata_name = f"InformationRegister.{register_name}"
            id = generate_uuid(metadata_name)
            new_metadata = f'{new_metadata}		<Metadata name="{metadata_name}" id="{id}" configVersion="{CONFIG_VERSION}">'

            for dim_name, dim_type in dimensions:
                if dim_name in STANDARD_DIMENSIONS:
                    continue
                meta_dim_name = f"{metadata_name}.Dimension.{dim_name}"
                id = generate_uuid(meta_dim_name)
                new_metadata = f'{new_metadata}\n			<Metadata name="{meta_dim_name}" id="{id}"/>'

            new_metadata = f"{new_metadata}\n		</Metadata>\n"

        for index, line in enumerate(template):
            if "{NEW_METADATA}" in line:
                template[index] = template[index].replace(
                    "{NEW_METADATA}", new_metadata
                )

        with open(f"{DIR}/ConfigDumpInfo.xml", "wt") as f_result:
            f_result.writelines(template)


def create_catalogs(catalogs):
    """Создает файлы справочников в каталоге Catalogs"""

    with open(f"{DIR}/{TEMPLATES}/catalogs/catalog.xml", "r") as f:
        template_catalog = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/catalogs/attr_string.xml", "r") as f:
        attr_string = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/catalogs/attr_decimal.xml", "r") as f:
        attr_number = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/catalogs/attr_date.xml", "r") as f:
        attr_date = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/catalogs/attr_boolean.xml", "r") as f:
        attr_boolean = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/catalogs/attr_ref.xml", "r") as f:
        attr_ref = "".join(f.readlines())

    for catalog_name, attributes in catalogs.items():
        file_text = template_catalog.replace("{METADATA_NAME}", catalog_name)

        id = CACHE_ID.get(f"Catalog.{catalog_name}")
        file_text = file_text.replace("{METADATA_ID}", id)

        file_text = file_text.replace("{O_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{O_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{R_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{R_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{S_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{S_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{L_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{L_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{M_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{M_VALUE_ID}", generate_uuid())

        attributes_text = ""

        for attr_name, attr_type in attributes:
            if attr_name in STANDARD_ATTRIBUTES:
                continue

            meta_attr_name = f"Catalog.{catalog_name}.Attribute.{attr_name}"
            id = CACHE_ID.get(meta_attr_name)

            if attr_type == "Строка":
                attr_text = attr_string.replace("{ATTR_ID}", id).replace(
                    "{ATTR_NAME}", attr_name
                )
            elif attr_type == "Число":
                attr_text = attr_number.replace("{ATTR_ID}", id).replace(
                    "{ATTR_NAME}", attr_name
                )
            elif attr_type == "Дата":
                attr_text = attr_date.replace("{ATTR_ID}", id).replace(
                    "{ATTR_NAME}", attr_name
                )
            elif attr_type == "Булево":
                attr_text = attr_boolean.replace("{ATTR_ID}", id).replace(
                    "{ATTR_NAME}", attr_name
                )
            elif attr_type.lower().startswith("справочник"):
                attr_text = (
                    attr_ref.replace("{ATTR_ID}", id)
                    .replace("{ATTR_NAME}", attr_name)
                    .replace("{CATALOG_NAME}", attr_type.split(".")[1])
                )
            else:
                continue

            attributes_text = f"{attributes_text}\n{attr_text}"

        file_text = file_text.replace("{ATTRIBUTES}", attributes_text)

        with open(f"{DIR}/Catalogs/{catalog_name}.xml", "wt") as f_result:
            f_result.write(file_text)


def create_registers(registers):
    """Создает файлы регистров сведений в каталоге InformationRegisters"""

    with open(f"{DIR}/{TEMPLATES}/registers/register.xml", "r") as f:
        template_register = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/registers/dim_string.xml", "r") as f:
        dim_string = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/registers/dim_decimal.xml", "r") as f:
        dim_number = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/registers/dim_date.xml", "r") as f:
        dim_date = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/registers/dim_boolean.xml", "r") as f:
        dim_boolean = "".join(f.readlines())

    with open(f"{DIR}/{TEMPLATES}/registers/dim_ref.xml", "r") as f:
        dim_ref = "".join(f.readlines())

    for register_name, dimensions in registers.items():
        file_text = template_register.replace("{METADATA_NAME}", register_name)

        id = CACHE_ID.get(f"InformationRegister.{register_name}")
        file_text = file_text.replace("{METADATA_ID}", id)

        file_text = file_text.replace("{R_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{R_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{M_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{M_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{S_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{S_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{L_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{L_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{RS_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{RS_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{RK_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{RK_VALUE_ID}", generate_uuid())
        file_text = file_text.replace("{RM_TYPE_ID}", generate_uuid())
        file_text = file_text.replace("{RM_VALUE_ID}", generate_uuid())

        dimensions_text = ""

        for dim_name, dim_type in dimensions:
            if dim_name in STANDARD_DIMENSIONS:
                continue

            meta_attr_name = f"InformationRegister.{register_name}.Dimension.{dim_name}"
            id = CACHE_ID.get(meta_attr_name)

            if dim_type == "Строка":
                dim_text = dim_string.replace("{DIM_ID}", id).replace(
                    "{DIM_NAME}", dim_name
                )
            elif dim_type == "Число":
                dim_text = dim_number.replace("{DIM_ID}", id).replace(
                    "{DIM_NAME}", dim_name
                )
            elif dim_type == "Дата":
                dim_text = dim_date.replace("{DIM_ID}", id).replace(
                    "{DIM_NAME}", dim_name
                )
            elif dim_type == "Булево":
                dim_text = dim_boolean.replace("{DIM_ID}", id).replace(
                    "{DIM_NAME}", dim_name
                )
            elif dim_type.lower().startswith("справочник"):
                dim_text = (
                    dim_ref.replace("{DIM_ID}", id)
                    .replace("{DIM_NAME}", dim_name)
                    .replace("{CATALOG_NAME}", dim_type.split(".")[1])
                )
            else:
                continue

            dimensions_text = f"{dimensions_text}\n{dim_text}"

        file_text = file_text.replace("{DIMENSIONS}", dimensions_text)

        with open(f"{DIR}/InformationRegisters/{register_name}.xml", "wt") as f_result:
            f_result.write(file_text)


def main():
    # Чтение схемы
    catalogs, registers = parse_metadata_csv()
    if not catalogs:
        print("CSV-файл пуст или не удалось его распарсить!")
        return

    # Configuration.xml
    create_config(catalogs, registers)

    # ConfigDumpInfo.xml
    create_config_dump(catalogs, registers)

    # Catalogs
    create_catalogs(catalogs)

    # InformationRegisters
    create_registers(registers)


if __name__ == "__main__":
    main()
