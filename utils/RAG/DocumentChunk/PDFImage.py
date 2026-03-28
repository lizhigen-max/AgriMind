#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：PDFImage.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

from pdf2docx import Converter
import os
import logging
from WordImage import word_extract_and_replace_images


def pdf_extract_and_replace_images(pdf_path, output_dir='extracted_content', is_ocr=False):
    """
    将PDF文件转换为Word文档  需要手动去处理文件，删除无关内容和无关图片

    参数:
    pdf_path (str): PDF文件路径
    output_folder (str): 输出文件夹路径(可选)
    """
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    # 设置输出路径
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(pdf_path)
        docx_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.docx')
    else:
        docx_path = os.path.splitext(pdf_path)[0] + '.docx'

    # 执行转换
    cv = Converter(pdf_path)
    cv.convert(docx_path, start=0, end=None)
    cv.close()

    logging.info(f"转换完成! Word文档已保存到: {docx_path}")

    processed_images, new_docx_path = word_extract_and_replace_images(docx_path, output_dir=output_dir, is_ocr=is_ocr)
    os.remove(docx_path)
    return processed_images, new_docx_path


# 使用示例
if __name__ == "__main__":
    pdf_file = "../Document/小天帮农使用手册-地块管理员端.pdf"  # 替换为你的PDF文件路径

    try:
        pdf_extract_and_replace_images(pdf_file, is_ocr=True)
    except Exception as e:
        print(f"转换失败: {str(e)}")