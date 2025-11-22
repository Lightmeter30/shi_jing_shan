from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
from utils.times import timer
from utils.logger_config import logger
import pdb

@timer
@csrf_exempt
def chat_to_ollama(request):
    '''
    Ollama模型聊天视图
    处理与Ollama模型的聊天请求
    {
	"model": "qwen2.5:72b",
	"messages": [{
		"role": "system",
		"content": "你是一个专业的博物馆导览员，负责向游客介绍展品，现在展品有声呐和太空站。\r\n            你的角色特点：\r\n            1. 专业且友好 \r\n            2. 使用简洁清晰的语言 \r\n            3. 能够引导游客参与互动 \r\n\r\n            请根据游客的输入，给出合适的回应。"
	}, {
		"role": "user",
		"content": "现在面前的是一个声呐，用户提问的问题是声呐是什么,请以一个精通声呐的专家身份回答"
	}],
	"stream": false
}
    '''
    # Ollama API配置
    OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"

    if request.method == 'POST':
        # pdb.set_trace()
        print(request.POST)
        print(request.body)
        print(request.GET)
        message = request.POST.get('messages', '')
        MODEL_NAME = request.POST.get('model', "qwen2.5:72b")
        print(type(message))
        print(message)

        # 构建请求数据
        data = json.loads(request.body)
        try:
            response = requests.post(OLLAMA_BASE_URL, json=data)
            response.raise_for_status()  # 检查请求是否成功
            ai_response = response.json()
            return JsonResponse(ai_response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API请求失败: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)