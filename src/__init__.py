"""CT 肝脏肿瘤分割（基础版 2D 闭环）的 Python 包。

整体功能：提供配置读取、预处理 / Dataset / 模型 / 训练 / 评估等模块的公共入口。
前后接口：上游是仓库根目录的 configs/*.yaml 与 data/*.json；下游由 scripts/ 与 src/ 下的命令行脚本 import。
用法：在仓库根目录执行 ``python -m src.train --fold 0`` 之类的命令，不要直接运行本文件。
"""
