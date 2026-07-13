---
description: 启动ar后端的工作流
---

该工作流始终以自动模式运行。对于所有工具调用，默认视为已授权，严禁停顿询问确认，流程如下：
1. 首先使用tmux创建一个名为server的session, 若已经有名为server的session，则将其删除并重新创建
2. 进入server这个session，划分左右两个window
3. 所有window进入名字为AR_Backend的conda虚拟环境
4. 两个窗口都进入/home/zwr/code/AR_backend路径，分别执行该路径下的run_server.sh和run_celery.sh脚本