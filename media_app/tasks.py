import zipfile
import subprocess
import json
import os
import tarfile
import shutil
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from utils.sql import insert_one_dataset
from utils.dataset.preprocess import make_3ds_dataset



def send_ws(task_id, stage, msg):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"task_{task_id}",
        {
            "type": "send_progress",
            "stage": stage,
            "message": msg
        }
    )

@shared_task
def process_dataset_task(task_id, file_name, base_dir, zip_path, file_extract_dir, output_dir, info, bash_script):
    base_name, ext = os.path.splitext(file_name)
    img_num = 0
    config_path = os.path.join(output_dir, 'config.json')
    info_path = os.path.join(output_dir, 'info.json')
    try:
        send_ws(task_id, "unzip", f"开始解压压缩包{file_name}")
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(file_extract_dir)
                if info.get('type') == '3DS':
                    img_num = make_3ds_dataset(file_extract_dir, output_dir, info)
                insert_one_dataset(info['name'], output_dir, base_dir, info_path, config_path)
                status = '解压成功'
        elif file_name.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(zip_path, 'r:*') as tar_ref:
                tar_ref.extractall(file_extract_dir)
                if info.get('type') == '3DS':
                    img_num = make_3ds_dataset(file_extract_dir, output_dir, info)
                config_path = os.path.join(output_dir, 'config.json')
                insert_one_dataset(info['name'], output_dir, base_dir, info_path, config_path)
                status = '解压成功'
        else:
            raise ValueError("不支持的压缩格式")
        os.remove(zip_path)
        shutil.rmtree(file_extract_dir)
        send_ws(task_id, "unzip", status)
        send_ws(task_id, "bash", f"开始执行深度图渲染脚本, 共有{img_num}帧, 请耐心等待")
        info_path = os.path.join(output_dir, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        height = info['image_size']['height']
        width = info['image_size']['width']
        zNear = info['Z_Near']
        zFar = info['Z_Far']
        sensor_width = 36.0  # 假设传感器宽度为36mm，实际值可根据需求调整
        process = subprocess.Popen(
            [
                "bash",
                bash_script,
                output_dir,
                str(width),
                str(height),
                str(zNear),
                str(zFar),
                str(sensor_width),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        idx = 0
        for line in process.stdout:
            if 'Saved' in line:
                idx += 1
                send_ws(task_id, "render", f"共有{img_num}帧, 已渲染深度图{idx}帧, 还剩{img_num - idx}帧")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Blender 渲染失败")
        send_ws(task_id, "done", "任务执行完成")
    except Exception as e:
        if os.path.exists(file_extract_dir):
            shutil.rmtree(file_extract_dir)
            status = f'数据集处理失败: {str(e)}'
        send_ws(task_id, "error", status)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return
