from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk, OCRResponse
from pathlib import Path
import json
import os
import time

# API配置
api_key = "jeJUYCFVsdLttyycs15a49fiiIjr0zNn"
client = Mistral(api_key=api_key)

# 路径配置
pdf_dir = Path(r"E:\毕业设计\数据库\论文数据\PDF")
output_dir = Path(r"E:\毕业设计\数据库\论文数据\TXT文本")

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

def get_text_content(ocr_response: OCRResponse) -> str:
    """从OCR响应中提取纯文本内容"""
    text_parts = []
    for page in ocr_response.pages:
        # 提取markdown中的纯文本部分（去除图片引用等）
        page_text = page.markdown
        # 简单处理：移除markdown图片链接格式
        import re
        page_text = re.sub(r'!\[.*?\]\(.*?\)', '', page_text)
        text_parts.append(page_text)
    
    return "\n\n".join(text_parts)

def process_pdf_file(pdf_path, output_format="txt"):
    print(f"处理文件: {pdf_path.name}")
    
    try:
        # 上传文件
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        
        # 获取签名URL
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # 处理OCR
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url), 
            model="mistral-ocr-latest", 
            include_image_base64=True
        )
        
        # 根据输出格式选择处理方式
        if output_format.lower() == "md":
            # 生成Markdown格式
            output_content = get_combined_markdown(pdf_response)
            output_file = output_dir / f"{pdf_path.stem}.md"
        else:
            # 生成TXT格式（纯文本）
            output_content = get_text_content(pdf_response)
            output_file = output_dir / f"{pdf_path.stem}.txt"
        
        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_content)
            
        print(f"已成功转换并保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"处理文件 {pdf_path.name} 时出错: {str(e)}")
        return False

def main():
    # 获取所有PDF文件
    pdf_files = list(pdf_dir.glob("*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"在 {pdf_dir} 中未找到PDF文件")
        return
    
    print(f"找到 {total_files} 个PDF文件，开始处理...")
    
    # 选择输出格式
    output_format = "txt"  # 可以修改为 "md" 如果需要输出markdown
    
    # 处理计数
    success_count = 0
    
    # 批量处理所有PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n处理进度: {i}/{total_files} ({i/total_files*100:.1f}%)")
        
        if process_pdf_file(pdf_file, output_format):
            success_count += 1
        
        # 避免API速率限制，每个请求后稍微暂停
        if i < total_files:
            print("等待2秒后处理下一个文件...")
            time.sleep(2)
    
    print(f"\n处理完成! 成功处理 {success_count}/{total_files} 个文件")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()