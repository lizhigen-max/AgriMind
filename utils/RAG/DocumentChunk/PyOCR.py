#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：PyOCR.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

# 安装依赖
# pip install pillow pytesseract
# 还需要安装Tesseract-OCR引擎：https://github.com/tesseract-ocr/tesseract

import pytesseract
from PIL import Image
import re
import os
import logging


# 配置Tesseract路径（Windows需要）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image_to_text(image_path, lang='chi_sim+eng'):
    """
    识别图片中的文字

    Args:
        image_path: 图片路径
        lang: 语言包，chi_sim(简体中文)+eng(英文)

    Returns:
        识别出的文本
    """
    try:
        # 打开图片
        img = Image.open(image_path)

        # 可选的图像预处理
        # img = preprocess_image(img)

        # 使用OCR识别文字
        text = pytesseract.image_to_string(img, lang=lang)

        return text
    except Exception as e:
        logging.info(f"识别失败: {e}")
        return ""


def preprocess_image(img):
    """图像预处理函数，提高识别率"""
    # 转换为灰度图
    if img.mode != 'L':
        img = img.convert('L')

    # 可以添加更多预处理步骤：
    # 1. 调整对比度
    # 2. 二值化
    # 3. 降噪
    # 4. 旋转校正

    return img


def clean_text(text):
    """清理和整理识别出的文本"""
    if not text:
        return ""

    # 1. 移除多余的空格和换行
    # 合并多个换行符
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # 移除行首行尾空格
    lines = [line.strip() for line in text.split('\n')]

    # 2. 过滤空行
    lines = [line for line in lines if line.strip()]

    # 3. 智能段落合并（可选）
    # 如果一行很短，可能不是段落结尾
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i > 0 and len(lines[i - 1]) > 30 and len(line) > 0:
            # 上一行较长，当前行非空，保持换行
            cleaned_lines.append(line)
            continue
        elif i > 0 and len(line) < 20 and not line.endswith(('。', '!', '?', '！', '？')):
            # 短行且不以句号结尾，可能是同一段落
            if cleaned_lines:
                cleaned_lines[-1] += " " + line
                continue
        cleaned_lines.append(line)

    return ' '.join(cleaned_lines)


def remove_chinese_spaces_improved(text):
    # 使用正则表达式
    # 匹配中文之间的空格，但不匹配以下情况：
    # 1. 空格前面是英文
    # 2. 空格后面是英文
    # 3. 空格前后是数字

    # 匹配中文（包括扩展区）
    chinese_char = r'[\u4e00-\u9fff\u3400-\u4dbf]'

    # 匹配中文标点
    chinese_punct = r'[，。！？；："\'（）【】《》]'

    # 中文或中文标点
    chinese_pattern = f'({chinese_char}|{chinese_punct})'

    # 匹配英文单词字符（字母、数字、下划线）
    english_word = r'[a-zA-Z0-9_]'

    # 构建正则表达式：
    # 匹配前面是中文或中文标点，后面是中文或中文标点的空格
    # 但不匹配前面或后面是英文的情况
    pattern = rf'(?<={chinese_pattern})\s+(?={chinese_pattern})'

    # 替换中文之间的空格
    result = re.sub(pattern, '', text)
    return result

def ocr_and_organize(image_path, output_txt=None, lang='chi_sim+eng'):
    """
    完整的OCR识别和整理流程

    Args:
        image_path: 图片路径
        output_txt: 输出文本文件路径
        lang: 语言
    """
    logging.info(f"正在处理: {image_path}")

    # 1. OCR识别
    raw_text = ocr_image_to_text(image_path, lang)
    logging.info("=== 原始识别结果 ===")
    logging.info(raw_text)

    # 2. 文本整理
    cleaned_text = clean_text(raw_text)
    cleaned_text = remove_chinese_spaces_improved(cleaned_text)
    prilogging.infont("\n=== 整理后的文本 ===")
    logging.info(cleaned_text)

    # 3. 保存到文件
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        prlogging.infoint(f"\n文本已保存到: {output_txt}")

    return cleaned_text


# 使用示例
if __name__ == "__main__":
    # 单个图片处理
    result = ocr_and_organize(
        image_path=r"C:\Users\Zhigen\Desktop\112.png",
        output_txt=None,
        lang='chi_sim+eng'  # 中文+英文
    )

    # 批量处理
    # for img_file in os.listdir("images_folder"):
    #     if img_file.endswith(('.jpg', '.png', '.jpeg')):
    #         txt_file = f"output_{os.path.splitext(img_file)[0]}.txt"
    #         ocr_and_organize(
    #             image_path=os.path.join("images_folder", img_file),
    #             output_txt=txt_file
    #         )