import sqlite3
import json
from datetime import datetime
from django_project import settings
import os
from typing import List, Dict, Any
# import pandas as pd
from .logger_config import logger

DB_PATH = os.path.join(settings.BASE_DIR, 'AR_platform.db')

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

def delete_scence_by_name(name: str, db_path: str = DB_PATH) -> bool:
    """
    根据名称删除 SCENCE 表中的记录
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM SCENCE WHERE NAME = ?", (name,))
            conn.commit()
            logger.info(f"Deleted records with NAME '{name}' from SCENCE table.")
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error deleting records: {e}")
        return False

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