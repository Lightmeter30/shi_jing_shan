import sqlite3
import json
from datetime import datetime
from django_project import settings
import os
from typing import List, Dict, Any
# import pandas as pd
from .logger_config import logger
from media_app.models import *
from django.db import transaction

DB_PATH = os.path.join(settings.BASE_DIR, 'AR_platform.db')

def insert_one_dataset(name: str, abs_dataset_dir: str, base_dir: str, info_path: str, config_path: str) -> None:
    '''
    插入单条数据（原子性保证）
    '''
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(info_path, 'r') as f:
        info = json.load(f)

    with transaction.atomic():
        # 创建并保存 Dataset 实例
        dataset_instance = Dataset.objects.create(
            name=name,
            file_path=os.path.relpath(abs_dataset_dir, base_dir),
            info=info,
            config=config,
            old_config=config
        )

        # 遍历并保存 DatasetFile 实例
        for file in os.listdir(abs_dataset_dir):
            file_path = os.path.join(abs_dataset_dir, file)
            if os.path.isfile(file_path):
                if file.endswith('.json'):
                    continue
                file_type = 'object' if file.endswith('.obj') else 'ply' if file.endswith('.ply') else 'unknown'
                DatasetFile.objects.create(
                    dataset=dataset_instance,
                    name=file,
                    file_path=os.path.relpath(file_path, base_dir),
                    file_type=file_type
                )

def insert_sence_batch(records: List[Dict[str, Any]], da_path = DB_PATH) -> None:
    """
    批量插入记录到 SCENCE 表，支持自动递增主键。
    每条记录应包含: NAME, CONFIG (dict)，可选: CREATE_DATE, ID
    """
    try:
        with sqlite3.connect(da_path) as conn:
            cursor = conn.cursor()
            
            for record in records:
                name = record.get("NAME")
                create_date = record.get("CREATE_DATE", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                config = json.dumps(record.get("CONFIG", {}))

                if "ID" in record:
                    id_value = record["ID"]
                    cursor.execute("INSERT INTO SCENCE (ID, NAME, CREATE_DATE, CONFIG) VALUES (?, ?, ?, ?)",
                                   (id_value, name, create_date, config))
                else:
                    cursor.execute("INSERT INTO SCENCE (NAME, CREATE_DATE, CONFIG) VALUES (?, ?, ?)",
                                   (name, create_date, config))
            
            conn.commit()
            logger.info(f"Inserted {len(records)} records into SCENCE table.")
    except Exception as e:
        logger.error(f"Error inserting records: {e}")

def get_config_field(scence_id: int, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    读取指定 ID 的 CONFIG 字段并解析为 Python 字典
    """
    config_dict = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT CONFIG FROM SCENCE WHERE ID = ?", (scence_id,))
            row = cursor.fetchone()
            if row:
                config_dict = json.loads(row[0])
                logger.info(f"Retrieved config for SCENCE ID {scence_id}.")
            else:
                logger.warning(f"No record found for SCENCE ID {scence_id}.")
    except Exception as e:
        logger.error(f"Error retrieving config: {e}")
    return config_dict

def get_all_scences(da_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    获取 SCENCE 表的所有记录，返回字典列表
    """
    try:
        with sqlite3.connect(da_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ID, NAME, CREATE_DATE FROM SCENCE")
            rows = cursor.fetchall()
            
            # 将查询结果转换为字典列表
            result = []
            for row in rows:
                result.append({
                    'sceneName': row[1],
                    'scenceKey': str(row[0]),
                    # 'createDate': row[2]
                })
            
            logger.info("Retrieved all records from SCENCE table.")
            return result
    except Exception as e:
        logger.error(f"Error retrieving all scences: {e}")
        return []