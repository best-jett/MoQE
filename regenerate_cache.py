import os
import shutil
import logging
import torch
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import glob

def setup_logging():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cache_regeneration_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志将保存到: {log_file}")
    return logger

def process_and_cache_file(file_path, tokenizer, processed_cache_dir, max_length, min_words, max_words):
    logger = logging.getLogger(__name__)
    cache_file_name = f"{Path(file_path).stem}.pt"
    cache_path = os.path.join(processed_cache_dir, cache_file_name)
    meta_path = cache_path + ".meta"

    logger.info(f"开始处理: {os.path.basename(file_path)} -> 目标缓存: {os.path.basename(cache_path)}")
    
    processed_samples = []
    try:
        df = pd.read_parquet(file_path)
        if 'text' not in df.columns:
            logger.error(f"Parquet 文件 {file_path} 中未找到 'text' 列。跳过此文件。")
            return

        for text in tqdm(df['text'], desc=f"Tokenizing {os.path.basename(file_path)}", leave=False):
            if not isinstance(text, str) or not text.strip():
                continue

            word_count = len(text.split())
            if not (min_words <= word_count <= max_words):
                continue
            
            tokens = tokenizer(
                text,
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
            logger.info(f"成功缓存 {len(processed_samples)} 个样本到 {os.path.basename(cache_path)}")
        else:
            logger.warning(f"文件 {os.path.basename(file_path)} 未产生任何有效样本。")

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生严重错误: {e}", exc_info=True)


def main():
    """
    主函数，执行删除旧缓存和生成新缓存的全部流程。
    """
    logger = setup_logging()
    
    # --- 配置 ---
    data_dir = "/path/to/data"
    cache_dir = "/path/to/data_cache"
    tokenizer_path = "path/to/tokenizer"
    max_length = 512
    min_words = 100
    max_words = 1000
    # --- 结束配置 ---

    processed_cache_dir = os.path.join(cache_dir, "your_data_name")

    logger.info(f"检查并删除旧的缓存目录: {processed_cache_dir}")
    if os.path.exists(processed_cache_dir):
        try:
            shutil.rmtree(processed_cache_dir)
            logger.info("旧缓存目录已成功删除。")
        except OSError as e:
            logger.error(f"删除旧缓存失败: {e}", exc_info=True)
            return
    
    os.makedirs(processed_cache_dir, exist_ok=True)
    logger.info(f"已创建新的缓存目录: {processed_cache_dir}")

    logger.info(f"正在从 {tokenizer_path} 加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"加载分词器失败: {e}", exc_info=True)
        return

    source_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not source_files:
        logger.error(f"在 {data_dir} 中未找到任何 .parquet 文件。")
        return
    
    logger.info(f"找到 {len(source_files)} 个源文件进行处理: {[os.path.basename(f) for f in source_files]}")

    for file_path in source_files:
        process_and_cache_file(
            file_path,
            tokenizer,
            processed_cache_dir,
            max_length,
            min_words,
            max_words
        )
    
    logger.info("="*50)
    logger.info(" 所有缓存文件已成功重新生成！")
    logger.info("="*50)


if __name__ == "__main__":
    main() 
