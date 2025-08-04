import os
import shutil
import logging
import re
import torch
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import glob

# ---------- 日志 ----------
def setup_logging():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cache_regeneration_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志将保存到: {log_file}")
    return logger

# ---------- 文本清洗 ----------
def clean_text(text: str) -> str:
    """
    - 移除空行
    - 移除 HTML 标签
    - 移除 URL
    - 移除包含重复字符的冗余行
    返回清洗后的整段文本
    """
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:                       # 空行
            continue
        # 移除 HTML 标签
        line = re.sub(r"<[^>]+>", "", line)
        # 移除 URL
        line = re.sub(r"http\S+|www\.\S+", "", line, flags=re.IGNORECASE)
        # 移除重复字符行（同一字符连续出现≥3次）
        if re.search(r"(.)\1{2,}", line):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def process_and_cache_file(file_path, tokenizer, processed_cache_dir, max_length, min_words, max_words):
    logger = logging.getLogger(__name__)
    cache_file = f"{Path(file_path).stem}.pt"
    cache_path = os.path.join(processed_cache_dir, cache_file)
    meta_path = cache_path + ".meta"

    logger.info(f"开始处理: {os.path.basename(file_path)} -> {cache_file}")
    processed_samples = []

    try:
        df = pd.read_parquet(file_path)
        if 'text' not in df.columns:
            logger.error(f"{file_path} 缺少 'text' 列，跳过。")
            return

        for raw_text in tqdm(df['text'], desc=f"Processing {os.path.basename(file_path)}", leave=False):
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            word_cnt = len(cleaned.split())
            if not (min_words <= word_cnt <= max_words):
                continue

            tokens = tokenizer(
                cleaned,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = tokens['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            processed_samples.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)
            })

        if processed_samples:
            torch.save(processed_samples, cache_path)
            with open(meta_path, 'w') as f:
                json.dump({'count': len(processed_samples)}, f)
            logger.info(f"已缓存 {len(processed_samples)} 条样本到 {cache_file}")
        else:
            logger.warning(f"{file_path} 未产生有效样本。")

    except Exception as e:
        logger.error(f"处理 {file_path} 时出错: {e}", exc_info=True)


# ---------- 主入口 ----------
def main():
    logger = setup_logging()

    # ------------- 用户配置 -------------
    data_dir        = "/path/to/data"
    cache_dir       = "/path/to/data_cache"
    tokenizer_path  = "path/to/tokenizer"
    max_length      = 2048          # 统一截断长度
    min_words       = 10            # 可根据需要调整
    max_words       = 10000
    # ------------- 配置结束 -------------

    processed_cache_dir = os.path.join(cache_dir, "cleaned_2048")
    if os.path.exists(processed_cache_dir):
        shutil.rmtree(processed_cache_dir)
    os.makedirs(processed_cache_dir, exist_ok=True)

    logger.info("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    src_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not src_files:
        logger.error("未找到任何 .parquet 文件。")
        return

    for fp in src_files:
        process_and_cache_file(fp, tokenizer, processed_cache_dir, max_length, min_words, max_words)

    logger.info("=" * 50)
    logger.info("所有缓存已重新生成完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
