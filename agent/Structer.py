#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：Structer.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated
from dataclasses import dataclass
from datetime import datetime
import json
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from prompts import intentClassifier_template, intentClassifier_template2
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
from enum import Enum

class Intent(str, Enum):
    """识别用户意图,分析用户消息意图分类"""
    AGRONOMIST = "agronomist"  # 农事相关问题，比如作物的物候期、水肥管理、病虫害防治、农事操作
    ORDINARY = "ordinary"  # 不属于以上这几种问题，或者你无法把握的问题

class Agronomist(BaseModel):
    """若用户意图是agronomist（农事相关问题），则提取agronomist相关信息"""
    crop: Optional[str] = Field(None, description="作物类型，比如：水稻、苹果、葡萄", max_length=10)
    behavior: str = Field(description="农事类型、行为，比如作物的物候期、水肥管理、病虫害防治、农事操作")
    confidence: float = Field(description="对根据用户消息提取到的农事类型进行置信度打分", ge=0, le=1)

class Ordinary(BaseModel):
    """若用户意图是ordinary（未知问题/其他问题），则提取ordinary相关信息"""

class StructOutput(BaseModel):
    """根据用户输入，提取意图以及关键信息"""
    intent: Intent = Field(description="识别用户意图,分析用户消息意图分类")
    confidence: float = Field(description="对根据用户消息提取到的用户意图进行置信度打分", ge=0, le=1)
    reason: str = Field(description="分析根据用户消息提取到的用户意图的原因", max_length=100)
    address: str = Field('温宿', description="根据用户消息提取地点信息")
    time_target: str = Field(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), description="根据用户消息提取时间信息")
    has_addr: bool = Field(description="用户消息中是否提取到了地点信息")
    has_time: bool = Field(description="用户消息中是否提取到了时间信息")
    query_strengthen: str = Field(description="query增强，明确用户意图，对用户问题进行增强")

    agronomist: Optional[Agronomist] = Field(None, description="若用户意图是agronomist，则提取agronomist相关信息")
    ordinary: Optional[Ordinary] = Field(None, description="若用户意图是ordinary，则提取ordinary相关信息")