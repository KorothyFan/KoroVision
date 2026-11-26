# -*- coding: utf-8 -*-
"""
KoroVision PythonAnywhere WSGI 入口文件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在 PythonAnywhere Web App 配置中：
  - Source code 目录: /home/<你的用户名>/KoroVision-Server
  - WSGI configuration file: /var/www/<用户名>_pythonanywhere_com_wsgi.py
  - 将本文件内容复制到上面的 WSGI 配置文件中
"""

import os
import sys

# ── 设置项目路径（把 YOUR_USERNAME 换成你的 PythonAnywhere 用户名） ──
project_home = '/home/YOUR_USERNAME/KoroVision-Server'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)

from server import app as application, load_model

print('[WSGI] Loading KoroVision model...')
load_model()
print('[WSGI] Model loaded, app ready')
