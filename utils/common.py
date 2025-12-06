#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Date     : 2025/12/6 00:18 
@Author   : ArcRay 
@FileName : common.py
@Brief    : 一些公共函数
"""

import os
import yaml

def get_class(yaml_file):
    if not os.path.exists(yaml_file):
        raise f"{yaml_file} does not exist!!"
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data
