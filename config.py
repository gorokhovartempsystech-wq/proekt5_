"""Пути и имена колонок датасета."""

import os

# Путь к исходному датасету: положите файл в data/ или задайте ER_DATA.
DATA = os.environ.get("ER_DATA", os.path.join("data", "student_relations.parquet"))

# рабочая папка для промежуточных артефактов
WORK = os.environ.get("ER_WORK", "work")

# колонки датасета
COL_ID = "record_id"
COL_COUNTRY = "country"
COL_DATE = "source_snapshot_date"
COL_REL_KIND = "relation_kind"
COL_REL_ROLE = "relation_role"
COL_PARTY_TYPE = "party_type"
COL_COMPANY_ID = "company_public_id"
COL_COMPANY_NAME = "company_name_norm"
COL_GOLD = "party_public_id"  # частично заполненная разметка - 3% базы
COL_NAME = "party_name"  # главное поле для матчинга
COL_SHARE = "ownership_share_pct"

RANDOM_STATE = 42

os.makedirs(WORK, exist_ok=True)
