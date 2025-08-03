import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, AutoConfig, GPTQConfig
import logging
import json
import argparse
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import time
import signal
import glob
import random
from tqdm import tqdm
import re
from pathlib import Path
import traceback
import gguf
import math
import itertools
import tempfile
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import gc
import types
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import psutil
import torch.nn.parallel
from torch import amp
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import sys

DATASET_PATH_FOR_PERPLEXITY = 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 配置日志
def setup_logging(log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'logs')
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置日志格式
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

# 初始化日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def check_gpu_memory():

    if not torch.cuda.is_available():
        return []
    
    gpu_memory_info = []
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
        free_memory_gb = free_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_properties(i).name
        gpu_memory_info.append({
            'index': i,
            'name': gpu_name,
            'free_memory_gb': free_memory_gb,
            'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        })
    
    # 按可用内存从大到小排序
    gpu_memory_info.sort(key=lambda x: x['free_memory_gb'], reverse=True)
    return gpu_memory_info

def allocate_models_to_gpus(model_configs, gpu_threshold_gb=4.0):
    if not torch.cuda.is_available():
        logger.warning("未检测到可用的GPU，所有模型将使用CPU")
        for config in model_configs:
            config['device_map'] = 'cpu'
        return model_configs
    
    # 获取GPU内存信息
    gpu_info = check_gpu_memory()
    if not gpu_info:
        logger.warning("无法获取GPU内存信息，将使用默认设置")
        return model_configs
    
    logger.info(f"检测到 {len(gpu_info)} 个可用GPU:")
    for gpu in gpu_info:
        logger.info(f"GPU {gpu['index']}: {gpu['name']}, 可用内存: {gpu['free_memory_gb']:.2f}GB / 总内存: {gpu['total_memory_gb']:.2f}GB")
    
    # 估算模型大小
    model_sizes = []
    for config in model_configs:
        # 基于模型路径和类型估算大小
        size_gb = 0
        path_lower = config['path'].lower()
        
        if config.get('load_in_4bit', False):
            size_factor = 0.25  # 4bit量化模型约为原始大小的1/4
        elif config.get('load_in_8bit', False):
            size_factor = 0.5   # 8bit量化模型约为原始大小的1/2
        else:
            size_factor = 1.0   # 全精度模型
            
        if '70b' in path_lower:
            base_size_gb = 70.0 * 2  # 参数数量(70B) * 每参数字节数(FP16约2字节)
        elif '13b' in path_lower:
            base_size_gb = 13.0 * 2
        elif '8b' in path_lower:
            base_size_gb = 8.0 * 2
        elif '7b' in path_lower:
            base_size_gb = 7.0 * 2
        elif '3b' in path_lower or '3-b' in path_lower:
            base_size_gb = 3.0 * 2  # 3B模型
        else:
            base_size_gb = 7.0 * 2  # 默认假设为7B
        
        overhead_factor = 1.3 
        size_gb = base_size_gb * size_factor * overhead_factor
        
        if path_lower.endswith('.gguf'):
            size_gb *= 0.8  # GGUF格式通常比标准模型小约20%
        
        model_name = os.path.basename(config['path'])
        model_sizes.append({
            'config': config,
            'estimated_size_gb': size_gb,
            'name': model_name
        })
        
        logger.info(f"模型 {model_name} 估计大小: {size_gb:.2f}GB")

    model_sizes.sort(key=lambda x: x['estimated_size_gb'], reverse=True)
    
    # 为每个模型分配GPU
    gpu_usage = [0.0] * len(gpu_info)  # 跟踪每个GPU的使用情况
    auto_allocation_models = []  # 跟踪设置为auto的模型
    
    for model in model_sizes:
        config = model['config']
        size_gb = model['estimated_size_gb']
        
        if size_gb > gpu_threshold_gb and len(gpu_info) > 1:
            # 大模型使用模型并行（多GPU）
            logger.info(f"模型 {model['name']} (估计{size_gb:.1f}GB) 将使用模型并行跨多个GPU")
            config['device_map'] = 'auto'  # 让transformers自动分配
            
            # 假设auto模式会在所有GPU上均匀分配
            auto_allocation_models.append(model)
            # 估计每个GPU上会分配的内存
            per_gpu_size = size_gb / len(gpu_info)
            for i in range(len(gpu_usage)):
                gpu_usage[i] += per_gpu_size
        else:
            # 为小模型找到最佳的单个GPU
            best_gpu_idx = 0
            min_usage = float('inf')
            
            for i, usage in enumerate(gpu_usage):
                if usage + size_gb < gpu_info[i]['free_memory_gb'] and usage < min_usage:
                    min_usage = usage
                    best_gpu_idx = i
            
            actual_gpu_idx = gpu_info[best_gpu_idx]['index']
            config['device_map'] = f'cuda:{actual_gpu_idx}'
            gpu_usage[best_gpu_idx] += size_gb
            
            logger.info(f"模型 {model['name']} (估计{size_gb:.1f}GB) 分配到 GPU {actual_gpu_idx}")
    
    # 打印最终分配情况
    logger.info("最终GPU分配情况:")
    for i, usage in enumerate(gpu_usage):
        actual_idx = gpu_info[i]['index']
        logger.info(f"GPU {actual_idx}: 估计使用 {usage:.2f}GB / {gpu_info[i]['free_memory_gb']:.2f}GB 可用")
    
    # 如果有auto分配模型，给出额外说明
    if auto_allocation_models:
        logger.info("\n自动分配模型详情:")
        total_auto_size = sum(model['estimated_size_gb'] for model in auto_allocation_models)
        logger.info(f"- 共有 {len(auto_allocation_models)} 个模型设置为'auto'，总估计大小: {total_auto_size:.2f}GB")
        logger.info(f"- 这些模型将由transformers库自动分配到可用GPU")
        logger.info(f"- 实际使用可能与估计不同，取决于transformers的分配策略")
        for model in auto_allocation_models:
            logger.info(f"  - {model['name']}: 估计 {model['estimated_size_gb']:.2f}GB (设置为'auto')")
    
    return [model['config'] for model in model_sizes]

UNIFIED_TOKENIZER_PATH = "path/to/embedding"
UNIFIED_TOKENIZER_FILE = os.path.join(UNIFIED_TOKENIZER_PATH, "tokenizer.json")

# 检查统一分词器文件是否存在
if os.path.exists(UNIFIED_TOKENIZER_FILE):
    logger.info(f"找到统一分词器文件: {UNIFIED_TOKENIZER_FILE}")
else:
    logger.warning(f"统一分词器文件不存在: {UNIFIED_TOKENIZER_FILE}")
    logger.warning("请确保分词器文件存在，否则可能导致加载失败")

class ExpertModel(nn.Module):
    def __init__(
        self, 
        model_path: str, 
        device_map: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        disable_marlin: bool = False,
        tokenizer_path: str = "path/to/embedding"
    ):
        super().__init__()
        self.model_path = model_path
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.disable_marlin = disable_marlin
        self.model = None
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        
        # 检查是否是GGUF模型
        self.is_gguf = model_path.lower().endswith('.gguf')
        if self.is_gguf:
            logger.info(f"检测到GGUF模型 {model_path}")
        
    def load(self):
        """加载模型和分词器"""
        if self.model is not None:
            logger.info(f"模型 {self.model_path} 已经加载，跳过。")
            return

        logger.info(f"正在加载专家模型: {self.model_path}")
        
        try:
            logger.info(f"加载统一分词器: {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, use_fast=True, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("分词器加载成功。")

            # 加载模型
            if self.is_gguf:
                logger.info(f"尝试加载 GGUF 模型: {self.model_path}")
                
                # 确保GGUF文件存在
                if not os.path.exists(self.model_path):
                    logger.error(f"GGUF文件不存在: {self.model_path}")
                    return False
                
                try:
                    model_dir = os.path.dirname(self.model_path)
                    gguf_filename = os.path.basename(self.model_path)
                    logger.info(f"使用GGUF加载方法 - 目录: {model_dir}, 文件: {gguf_filename}")
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        gguf_file=gguf_filename,
                        device_map=self.device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info(f"GGUF模型加载成功")
                    return True
                except Exception as e:
                    logger.error(f"GGUF模型加载错误: {str(e)}")
                    logger.info("尝试使用备用加载方法...")
                    
                    try:
                        # 备用加载方法：确保目录和文件名正确分离
                        from pathlib import Path
                        path = Path(self.model_path)
                        model_dir = str(path.parent)
                        gguf_filename = path.name
                        
                        logger.info(f"备用加载方法 - 目录: {model_dir}, 文件: {gguf_filename}")
                        
                        auto_gptq_module = sys.modules.pop("auto_gptq", None)
                        if auto_gptq_module:
                            logger.info("已临时移除 auto_gptq 以强制使用 Optimum 后端。")
                        
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_dir,
                                gguf_file=gguf_filename,
                                device_map=self.device_map,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True
                            )
                        finally:
                            if auto_gptq_module:
                                sys.modules["auto_gptq"] = auto_gptq_module
                                logger.info("已恢复 auto_gptq 模块。")
                        
                        logger.info(f"使用备用方法成功加载GGUF模型")
                        return True
                    except Exception as inner_e:
                        logger.error(f"备用加载方法也失败了: {str(inner_e)}")
                        logger.debug(traceback.format_exc())
                        return False
            else:
                model_load_path = self.model_path
                # 对于safetensors或bin文件，都使用其父目录进行加载
                if model_load_path.lower().endswith(('.safetensors', '.bin')):
                    model_load_path = os.path.dirname(model_load_path)

                model_kwargs = {
                    'device_map': self.device_map,
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True
                }

                quantization_config_to_use = None
                config = None
                try:
                    config = AutoConfig.from_pretrained(model_load_path, trust_remote_code=True)
                except Exception as e:
                    logger.warning(f"无法预加载模型配置: {e}。将继续尝试直接加载。")

                if config and hasattr(config, 'quantization_config') and config.quantization_config:
                    quant_method = config.quantization_config.get("quant_method", "").lower()
                    if ("gptq" in quant_method or "compressed-tensors" in quant_method) and self.disable_marlin:
                        logger.info(f"检测到GPTQ类模型({quant_method})，应用 'disable_marlin=True' 逻辑。")
                        quantize_config_path = os.path.join(model_load_path, "quantize_config.json")
                        if os.path.exists(quantize_config_path):
                            logger.info(f"从 {quantize_config_path} 加载详细GPTQ配置。")
                            with open(quantize_config_path, 'r') as f:
                                quant_config_data = json.load(f)
                            
                            quantization_config_to_use = GPTQConfig(
                                bits=quant_config_data.get("bits"),
                                group_size=quant_config_data.get("group_size", -1),
                                desc_act=quant_config_data.get("desc_act", False),
                                sym=quant_config_data.get("sym", True),
                                disable_marlin=True
                            )
                            logger.info("已通过 quantize_config.json 创建并覆盖GPTQConfig。")
                        else:
                            logger.warning("未找到 quantize_config.json，将尝试从主config构建GPTQConfig。")
                            from_dict_config = config.quantization_config
                            quantization_config_to_use = GPTQConfig.from_dict(from_dict_config)
                            quantization_config_to_use.disable_marlin = True
                            logger.info("已通过主config创建并覆盖GPTQConfig。")

                if quantization_config_to_use is None:
                    if self.load_in_4bit or self.load_in_8bit:
                        logger.info(f"应用BitsAndBytes量化: 4-bit={self.load_in_4bit}, 8-bit={self.load_in_8bit}")
                        quantization_config_to_use = BitsAndBytesConfig(
                            load_in_4bit=self.load_in_4bit,
                            load_in_8bit=self.load_in_8bit,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )
                
                if quantization_config_to_use is not None:
                    model_kwargs['quantization_config'] = quantization_config_to_use
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_load_path,
                    **model_kwargs
                )
                
                logger.info(f"标准/safetensors模型加载成功")
                return True
                
        except Exception as e:
            logger.error(f"加载模型或分词器时发生严重错误: {self.model_path}")
            logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
            return False
    
    def get_embedding_dim(self) -> int:
        if self.model is None:
            self.load()
        
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, "dim"):
                return self.model.config.dim
            elif hasattr(self.model.config, "d_model"):
                return self.model.config.d_model
        
        if hasattr(self.model, "get_input_embeddings"):
            embedding = self.model.get_input_embeddings()
            return embedding.embedding_dim
        
        logger.warning(f"无法确定模型嵌入维度，使用默认值4096")
        return 4096
    
    def get_vocabulary_size(self) -> int:
        if self.tokenizer is None:
            if self.is_gguf:
                try:
                    tokenizer_path = "path/to/tokenizer"  
                    logger.info(f"加载统一分词器以获取词汇表大小: {tokenizer_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        use_fast=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.error(f"加载统一分词器失败: {str(e)}")
                    return 32000  # 默认值
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path, 
                        use_fast=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.error(f"加载分词器失败: {str(e)}")
                    return 32000  # 默认值
        
        return len(self.tokenizer)

class MoERouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=3, num_heads=4, 
                 transformer_layers=2, dynamic_k=False, min_k=1, complexity_factor=0.7,
                 dropout=0.1, gating_temperature=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, self.num_experts)  # 默认top_k值
        self.num_heads = num_heads
        self.dynamic_k = dynamic_k 
        self.min_k = min_k  
        self.complexity_factor = complexity_factor  # 复杂度系数
        self.gating_temperature = gating_temperature  # 路由软化温度参数
        
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers
        )
        
        # 多头自注意力
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # MLP
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_experts)
        )
        
        # 用于计算输入复杂度的网络
        if self.dynamic_k:
            self.complexity_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()  # 输出0-1之间的复杂度分数
            )
        
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.register_buffer('expert_usage_counts', torch.zeros(num_experts))
        self.ema_alpha = 0.8  
        self.balance_strength = 0.8  
        
        self.use_noise = True
        if self.use_noise:
            self.noise_epsilon = nn.Parameter(torch.ones(1) * 1.0)  
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    
    def _calculate_dynamic_k(self, x):
        complexity_scores = self.complexity_net(x)  # [B, S, 1]
        avg_complexity = complexity_scores.mean(dim=1)  # [B, 1]
        k_range = self.top_k - self.min_k
        dynamic_k_float = self.min_k + k_range * avg_complexity
        dynamic_k = torch.clamp(dynamic_k_float.round().int(), 
                               min=self.min_k, 
                               max=self.top_k)
        
        return dynamic_k.squeeze(-1)  # [B]
    
    def forward(self, x, attention_mask=None):
        B, S, D = x.shape
        if hasattr(self, 'context_encoder'):
            if attention_mask is not None:
                transformer_mask = attention_mask.bool()
                context_encoder_dtype = next(self.context_encoder.parameters()).dtype
                x_encoder = x.to(context_encoder_dtype)
                x = self.context_encoder(x_encoder, src_key_padding_mask=~transformer_mask)
            else:
                context_encoder_dtype = next(self.context_encoder.parameters()).dtype
                x_encoder = x.to(context_encoder_dtype)
                x = self.context_encoder(x_encoder)
        
        # 确保后续操作使用一致的数据类型
        x_dtype = x.dtype
        
        # Self-Attention
        Q = self.query(x).view(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        K = self.key(x).view(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        V = self.value(x).view(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.num_heads)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        context = self.proj(context)
        context = self.proj_drop(context)
        
        # Residual + Norm
        x = self.norm(x + context)
        
        # 如果启用动态k值，计算每个样本的k值
        if self.dynamic_k:
            # 确保复杂度网络输入类型一致
            complexity_net_dtype = next(self.complexity_net.parameters()).dtype
            if x.dtype != complexity_net_dtype:
                complexity_input = x.to(complexity_net_dtype)
            else:
                complexity_input = x
                
            batch_k_values = self._calculate_dynamic_k(complexity_input)  # [B]
            # 不再使用max_k，而是保持每个样本的独立k值
        else:
            batch_k_values = torch.full((B,), self.top_k, device=x.device, dtype=torch.long)
        
        # Flatten and score experts
        flat_x = x.view(B * S, D)
        router_logits = self.fc(flat_x)
        
        # 应用温度缩放以调整路由软硬度
        if self.gating_temperature != 1.0:
            router_logits = router_logits / self.gating_temperature
        
        # Usage statistics - 使用实际的k值而不是max_k
        with torch.no_grad():
            batch_size_flat = router_logits.size(0)
            # 为每个样本使用其实际的k值进行topk
            sample_logits = router_logits.view(B, S, -1)
            all_top_indices = []
            total_actual_k = 0
            
            for i in range(B):
                sample_k = int(batch_k_values[i].item())
                _, top_indices = torch.topk(sample_logits[i], k=sample_k, dim=-1)
                all_top_indices.append(top_indices.flatten())
                total_actual_k += sample_k * S
            
            # 合并所有top_indices用于统计
            if all_top_indices:
                combined_top_indices = torch.cat(all_top_indices)
                expert_counts = torch.bincount(combined_top_indices, minlength=self.num_experts).float()
                expert_counts /= total_actual_k
                if self.training:
                    self.expert_usage_counts.mul_(self.ema_alpha).add_(expert_counts * (1 - self.ema_alpha))
            else:
                expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        adjusted_bias = torch.log(self.expert_usage_counts.mul(self.num_experts / (self.expert_usage_counts.sum() + 1e-10)) + 1e-10)
        strength = self.balance_strength if self.training else self.balance_strength * 0.5
        adjusted_bias.mul_(-strength)
        
        if self.training and self.use_noise:
            noise_std = F.softplus(self.noise_epsilon)
            current_epoch = getattr(self, 'current_epoch', 0)
            max_epochs = 5
            decay_factor = max(0.2, 1.0 - current_epoch / max_epochs)
            noise = torch.randn_like(router_logits) * noise_std * decay_factor
            router_logits += noise
        
        router_logits += self.bias + adjusted_bias
        
        # 使用真正的动态k值进行路由
        if self.dynamic_k:
            all_routing_weights = []
            all_routing_indices = []
            max_k_in_batch = int(batch_k_values.max().item())
            
            sample_logits = router_logits.view(B, S, -1)  
            
            for i in range(B):
                sample_k = int(batch_k_values[i].item())
                weights, indices = torch.topk(sample_logits[i], k=sample_k, dim=-1)  
                weights = F.softmax(weights, dim=-1)
                
                # 如果需要padding到最大k值
                if sample_k < max_k_in_batch:
                    weights = F.pad(weights, (0, max_k_in_batch - sample_k), "constant", 0)
                    indices = F.pad(indices, (0, max_k_in_batch - sample_k), "constant", 0)
                
                all_routing_weights.append(weights)
                all_routing_indices.append(indices)
            
            routing_weights = torch.stack(all_routing_weights, dim=0)  # [B, S, max_k_in_batch]
            routing_indices = torch.stack(all_routing_indices, dim=0)  # [B, S, max_k_in_batch]

            flat_routing_weights = routing_weights.view(-1, max_k_in_batch)
            flat_routing_indices = routing_indices.view(-1, max_k_in_batch)
        else:
            flat_routing_weights, flat_routing_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            flat_routing_weights = F.softmax(flat_routing_weights, dim=-1)
        
        # 计算负载均衡损失 - 使用实际的k值
        P = F.softmax(router_logits, dim=-1).mean(0)
        if self.dynamic_k:
            # 使用实际的k值计算负载均衡损失
            load_balancing_loss = self.num_experts * (P * expert_counts).sum()
        else:
            # 固定k值的情况
            F_values = torch.bincount(flat_routing_indices.flatten(), minlength=self.num_experts).float()
            F_values /= (batch_size_flat * self.top_k)
            load_balancing_loss = self.num_experts * (P * F_values).sum()
        
        return flat_routing_weights, flat_routing_indices, load_balancing_loss

class MoEModel(nn.Module):
    def __init__(
        self,
        expert_configs: List[Dict],
        top_k: int = 3,
        device: str = 'cuda',
        loaded_experts: List[ExpertModel] = None,
        current_epoch: int = 0,
        embedding_source_path: str = "path/to/embedding",
        router_config: Dict = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_balance_loss_coef: float = 0.1,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        logger.info("开始初始化浅层共享MoEModel架构（仅共享Embedding层）...")
        
        self.num_experts = len(loaded_experts)
        self.top_k = min(top_k, self.num_experts)
        self.main_device = device
        self.load_balance_loss_coef = load_balance_loss_coef
        self.gradient_checkpointing = gradient_checkpointing
        
        if loaded_experts is None or len(loaded_experts) < 1:
            raise ValueError("必须提供至少一个专家模型的已加载实例")
        
        logger.info("正在为所有专家配置完整的模型网络...")
        self.experts = nn.ModuleList()
        
        # 记录删除专家模型embedding层前后的内存使用情况
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        
        for i, expert_wrapper in enumerate(loaded_experts):
            expert_model = expert_wrapper.model  # LlamaForCausalLM
            expert_device = next(expert_model.parameters()).device
            
            if self.gradient_checkpointing:
                if hasattr(expert_model, 'gradient_checkpointing_enable'):
                    expert_model.gradient_checkpointing_enable()
                    logger.info(f"已为专家 {i} ({os.path.basename(expert_wrapper.model_path)}) 启用梯度检查点。")
                else:
                    logger.warning(f"专家 {i} 不支持 gradient_checkpointing_enable 方法。")
            
            try:
                if hasattr(expert_model, 'model') and hasattr(expert_model.model, 'embed_tokens'):
                    # 保存原始形状用于验证
                    original_shape = expert_model.model.embed_tokens.weight.shape
                    # 删除embedding层
                    expert_model.model.embed_tokens = None
                    logger.info(f"已删除专家 {i} 的embedding层，原始形状: {original_shape}")
                else:
                    logger.warning(f"无法删除专家 {i} 的embedding层，模型结构不符合预期")
            except Exception as e:
                logger.warning(f"删除专家 {i} 的embedding层时出错: {str(e)}")
            
            self.experts.append(expert_model)
            logger.info(f"专家 {i} 已配置为完整模型（无embedding层），位于 {expert_device}")
        
        # 记录删除后的内存使用情况
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"删除专家模型embedding层前后的内存使用: {before_mem:.2f} GB -> {after_mem:.2f} GB")
            logger.info(f"节省了约 {before_mem - after_mem:.2f} GB 的GPU内存")
        
        # 加载非量化模型，仅用于提取Embedding层
        logger.info(f"从 {embedding_source_path} 加载非量化模型以提取Embedding层...")
        embedding_model = None
        try:
            logger.info(f"直接将embedding层加载到 {self.main_device} 设备")
            embedding_model = AutoModelForCausalLM.from_pretrained(
                embedding_source_path,
                device_map=self.main_device,  # 直接加载到GPU
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            # 提取Embedding层
            source_model = embedding_model.model  # LlamaModel
            self.shared_embed = source_model.embed_tokens  # 已经在GPU上，无需再转移
            self.embed_dim = self.shared_embed.embedding_dim
            
            # 获取词汇表大小和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_source_path)
            self.vocab_size = len(self.tokenizer)
            
            logger.info(f"已从非量化模型提取共享Embedding层，维度: {self.embed_dim}")
            
            # 释放非量化模型内存，但保留分词器
            del embedding_model
            del source_model
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("已释放非量化模型内存，但保留了分词器")
        except Exception as e:
            logger.error(f"从非量化模型提取Embedding层失败: {str(e)}")
            if embedding_model is not None:
                del embedding_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # 如果提取失败，尝试从第一个专家模型提取
            logger.info("尝试从第一个专家模型提取Embedding层...")
            source_expert = loaded_experts[0].model.model  # LlamaModel
            if hasattr(source_expert, 'embed_tokens') and source_expert.embed_tokens is not None:
                self.shared_embed = source_expert.embed_tokens.to(self.main_device)
                self.embed_dim = self.shared_embed.embedding_dim
                self.vocab_size = loaded_experts[0].get_vocabulary_size()
                # 使用第一个专家的分词器
                self.tokenizer = loaded_experts[0].tokenizer
            else:
                logger.critical("无法提取embedding层，专家模型的embedding层已被删除")
                raise ValueError("无法提取embedding层")
        
        # 初始化路由器（输入是Embedding输出）
        router_config = router_config or {}
        self.router = MoERouter(
            self.embed_dim, 
            self.num_experts, 
            self.top_k, 
            num_heads=router_config.get('num_heads', 4),
            transformer_layers=router_config.get('transformer_layers', 2),
            dynamic_k=router_config.get('dynamic_k', False),
            min_k=router_config.get('min_k', 1),
            complexity_factor=router_config.get('complexity_factor', 0.7),
            dropout=router_config.get('dropout', 0.1),
            gating_temperature=router_config.get('gating_temperature', 1.0)
        ).to(self.main_device).to(torch_dtype)  # 强制转换为目标精度以确保一致
        self.router.current_epoch = current_epoch
        logger.info("浅层共享MoE模型初始化完成：输入 -> 共享Embedding -> 路由器 -> 完整专家模型 -> 输出")
    
    def get_expert_layer_params(self, layer_indices: List[int]):
        params = []
        for expert in self.experts:
            # 兼容不同模型结构
            all_layers = expert.model.layers if hasattr(expert.model, 'layers') else getattr(expert, 'layers', [])
            num_layers = len(all_layers)
            for layer_idx in layer_indices:
                # 转换负数索引为正数索引
                actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
                if 0 <= actual_idx < num_layers:
                    params.extend(list(all_layers[actual_idx].parameters()))
        return params

    # 冻结专家模型
    def _freeze_experts(self, fine_tune_layers=0):
        for i, expert in enumerate(self.experts):
            for param in expert.parameters():
                param.requires_grad = False
        logger.info("参数冻结完成。")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        input_ids = input_ids.to(self.main_device)
        attention_mask = attention_mask.to(self.main_device) if attention_mask is not None else None
        
        # 通过共享Embedding
        embeds = self.shared_embed(input_ids)
        
        # 路由器在Embedding输出上运行 (传入完整序列以捕捉上下文)
        flat_routing_weights, flat_routing_indices, load_balancing_loss = self.router(embeds, attention_mask)
        
        # 获取实际使用的k值（可能是动态的）
        actual_k = flat_routing_weights.size(-1)
        routing_weights = flat_routing_weights.view(batch_size, seq_len, actual_k)
        routing_indices = flat_routing_indices.view(batch_size, seq_len, actual_k)
        
        # 聚合专家输出（逐个处理以节省内存）
        try:
            # 初始化最终的logits张量，用于累加结果
            final_logits = torch.zeros(
                batch_size, seq_len, self.vocab_size,
                device=self.main_device,
                dtype=embeds.dtype  # 确保与输入类型一致，支持FP16
            )

            # 展平路由结果以方便索引
            flat_routing_weights = routing_weights.view(-1, actual_k)
            flat_routing_indices = routing_indices.view(-1, actual_k)

            # 逐个专家进行处理
            for i in range(self.num_experts):
                # 找到所有被路由到当前专家i的token
                mask = (flat_routing_indices == i)

                # 如果没有任何token被路由到这个专家，则跳过
                if not torch.any(mask):
                    continue

                # 获取需要计算的token的行索引和在top-k中的列索引
                rows, cols = torch.nonzero(mask, as_tuple=True)

                # 运行专家模型获取其对所有token的输出
                expert = self.experts[i]
                expert_device = next(expert.parameters()).device
                
                expert_embeds = embeds.to(expert_device)
                expert_mask = attention_mask.to(expert_device) if attention_mask is not None else None
                
                expert_logits = expert(
                    inputs_embeds=expert_embeds,
                    attention_mask=expert_mask,
                    return_dict=True
                ).logits.to(self.main_device)
                flat_expert_logits = expert_logits.view(-1, self.vocab_size)

                selected_logits = flat_expert_logits[rows]

                # 从路由权重中提取对应的权重
                weights_for_expert = flat_routing_weights[mask]

                # 将权重应用于提取出的logits，并确保数据类型一致
                weighted_logits = selected_logits * weights_for_expert.unsqueeze(1).to(selected_logits.dtype)

                # 使用index_add_就地、高效地将加权后的logits累加到最终结果中
                final_logits.view(-1, self.vocab_size).index_add_(0, rows, weighted_logits.to(final_logits.dtype))

                # 清理内存，为下一个专家腾出空间
                del expert_logits, flat_expert_logits, selected_logits, weighted_logits, mask, rows, cols
                if str(self.main_device).startswith('cuda'):
                     torch.cuda.empty_cache()

            logits = final_logits

        except RuntimeError as e:
            logger.error(f"聚合专家输出时出错: {str(e)}")
            logger.error(f"形状信息 - routing_weights: {routing_weights.shape}, routing_indices: {routing_indices.shape}")
            logger.error(f"actual_k: {actual_k}, batch_size: {batch_size}, seq_len: {seq_len}, vocab_size: {self.vocab_size}")
            raise
        
        loss = None
        ce_loss = None 
        if labels is not None:
            labels = labels.to(self.main_device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss() 
            ce_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            # 计算损失统计信息用于监控
            with torch.no_grad():
                loss_stats = {
                    'mean': ce_loss.mean().item() if ce_loss.numel() > 1 else ce_loss.item(),
                    'std': ce_loss.std().item() if ce_loss.numel() > 1 else 0.0,
                    'min': ce_loss.min().item(),
                    'max': ce_loss.max().item()
                }
                # 记录损失分布信息（仅用于调试）
                if hasattr(self, '_loss_stats_count'):
                    self._loss_stats_count += 1
                else:
                    self._loss_stats_count = 1
            
            # 动态调整负载均衡损失
            with torch.no_grad():
                expert_usage = torch.bincount(routing_indices.view(-1), minlength=self.num_experts).float()
                usage_mean = expert_usage.mean()
                if usage_mean > 0:
                    usage_std = expert_usage.std() / usage_mean
                else:
                    usage_std = torch.tensor(0.0, device=embeds.device, dtype=embeds.dtype)
                # 计算专家使用率的统计信息
                with torch.no_grad():
                    expert_usage_stats = {
                        'total_experts': self.num_experts,
                        'active_experts': (expert_usage > 0).sum().item(),
                        'usage_entropy': -torch.sum(expert_usage * torch.log(expert_usage + 1e-8)).item(),
                        'max_usage': expert_usage.max().item(),
                        'min_usage': expert_usage.min().item()
                    }
                    # 记录专家使用模式（仅用于分析）
                    if hasattr(self, '_expert_usage_history'):
                        self._expert_usage_history.append(expert_usage_stats)
                    else:
                        self._expert_usage_history = [expert_usage_stats]
                
                # 动态调整负载均衡系数
                dynamic_coef = self.load_balance_loss_coef * (1.0 + usage_std)  # 线性增长而非指数
            
            # 应用损失调整和正则化
            with torch.no_grad():
                # 计算损失梯度范数用于监控
                if ce_loss.requires_grad:
                    loss_grad_norm = torch.norm(ce_loss.grad) if ce_loss.grad is not None else torch.tensor(0.0)
                else:
                    loss_grad_norm = torch.tensor(0.0)
                
                # 记录损失调整信息
                loss_adjustment_info = {
                    'original_loss': ce_loss.item(),
                    'gradient_norm': loss_grad_norm.item(),
                    'adjustment_factor': 0.1,
                    'timestamp': time.time()
                }
                
                # 存储损失调整历史（仅用于调试）
                if hasattr(self, '_loss_adjustment_history'):
                    self._loss_adjustment_history.append(loss_adjustment_info)
                else:
                    self._loss_adjustment_history = [loss_adjustment_info]

            # 计算包含负载均衡的最终总损失
            loss = ce_loss + dynamic_coef * load_balancing_loss
            
            # 计算损失组成分析
            with torch.no_grad():
                loss_components = {
                    'ce_loss_weight': ce_loss.item(),
                    'load_balancing_weight': (dynamic_coef * load_balancing_loss).item(),
                    'total_loss': loss.item(),
                    'ce_loss_ratio': (ce_loss / loss).item() if loss.item() != 0 else 0.0,
                    'balancing_ratio': ((dynamic_coef * load_balancing_loss) / loss).item() if loss.item() != 0 else 0.0
                }
                
                # 记录损失组成历史（用于分析训练稳定性）
                if hasattr(self, '_loss_components_history'):
                    self._loss_components_history.append(loss_components)
                else:
                    self._loss_components_history = [loss_components]
            
            if not hasattr(self, '_forward_debug_printed'):
                # 困惑度应基于交叉熵损失计算
                perplexity = math.exp(ce_loss.item()) if ce_loss.item() > 0 and ce_loss.item() < 100 else float('inf')
                logger.info(f"浅层共享MoE首次前向传播 - 损失: {loss.item():.4f} (CE: {ce_loss.item():.4f} | Bal: {load_balancing_loss.item():.4f} | DynCoef: {dynamic_coef:.4f}), 困惑度: {perplexity:.2f}")
                self._forward_debug_printed = True
        
        return {
            "loss": loss, 
            "logits": logits,
            "routing_weights": routing_weights,
            "routing_indices": routing_indices,
            "load_balancing_loss": load_balancing_loss,
            "ce_loss": ce_loss 
        }

class TrainingState:
    def __init__(self):
        self.interrupted = False
        self.best_loss = float('inf')
        self.best_perplexity = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0
        self.last_save_time = time.time() # 跟踪上一次保存的时间
    
    def _signal_handler(self, sig, frame):
        logger.info(f"接收到信号 {sig}，准备安全停止训练...")
        self.interrupted = True
    
    def update(self, epoch, train_loss, val_loss, val_perplexity):
        """更新训练状态"""
        self.current_epoch = epoch
        if val_perplexity < self.best_perplexity:
            self.best_loss = val_loss
            self.best_perplexity = val_perplexity
            self.best_epoch = epoch
    
    def get_state(self):
        return {
            'interrupted': self.interrupted,
            'best_loss': self.best_loss,
            'best_perplexity': self.best_perplexity,
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch,
            'last_save_time': self.last_save_time
        }
    
    def load_state(self, state_dict):
        self.interrupted = state_dict.get('interrupted', False)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.best_perplexity = state_dict.get('best_perplexity', float('inf'))
        self.best_epoch = state_dict.get('best_epoch', -1)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.last_save_time = state_dict.get('last_save_time', time.time())
    
    def is_best_model(self):
        return self.current_epoch == self.best_epoch

def extract_unfrozen_parameters(model):
    unfrozen_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen_state_dict[name] = param.data.clone()
    
    # 添加模型结构信息
    unfrozen_state_dict['_model_structure'] = {
        'num_experts': model.num_experts,
        'top_k': model.top_k,
        'embed_dim': model.embed_dim,
    }
    
    logger.info(f"提取了 {len(unfrozen_state_dict)-1} 个解冻参数")
    return unfrozen_state_dict

def save_full_checkpoint(model, optimizer, scheduler, training_state, checkpoint_path, logger):
    temp_checkpoint_path = checkpoint_path + ".tmp"
    try:
        # 只保存可训练的参数
        trainable_params = extract_unfrozen_parameters(model)
        checkpoint = {
            'model_state_dict': trainable_params,
            'optimizer_state_dict': optimizer.state_dict(),
            'training_state': training_state.get_state()
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save to a temporary file first
        torch.save(checkpoint, temp_checkpoint_path)
        
        # Atomically rename the file
        os.rename(temp_checkpoint_path, checkpoint_path)
        
        logger.info(f"仅包含可训练参数的检查点已保存到: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"保存检查点到 {checkpoint_path} 失败: {e}")
        # Clean up the temporary file if it exists
        if os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)
        return False

class WikipediaDataset(Dataset):
    def __init__(self, wiki_dir, tokenizer, max_length=512, min_words=100, max_words=1000, 
                 cache_dir=None, seed=42, languages=None, preload_data=False, max_samples=None, file_prefix: Optional[str] = None):
        import psutil
        self.wiki_dir = wiki_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
        self.seed = seed
        self.languages = languages
        self.file_prefix = file_prefix
        
        # 创建用于存放预处理后文件的缓存目录
        self.processed_cache_dir = os.path.join(cache_dir, "wikipedia_preprocessed")
        os.makedirs(self.processed_cache_dir, exist_ok=True)
        
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.file_metadata = self._initialize_data()
        if max_samples is not None and max_samples > 0:
            accumulated = 0
            limited_metadata = []
            for meta in self.file_metadata:
                if accumulated >= max_samples:
                    break
                count = min(meta['count'], max_samples - accumulated)
                limited_metadata.append({'path': meta['path'], 'count': count})
                accumulated += count
            self.file_metadata = limited_metadata
            logger.info(f'Limited dataset to {accumulated} samples for faster training.')
        
        # 计算累积样本数，用于快速索引
        if not self.file_metadata:
            self.cumulative_sizes = np.array([0])
        else:
            counts = [meta['count'] for meta in self.file_metadata]
            self.cumulative_sizes = np.cumsum([0] + counts)
        
        total_samples = self.cumulative_sizes[-1]
        logger.info(f"Wikipedia数据集初始化完成，共 {len(self.file_metadata)} 个文件，{total_samples} 个样本")
        if total_samples == 0:
             logger.warning("数据集中没有找到任何样本，训练可能会失败")

        self.preload_data = preload_data
        self.data_cache = {}  # Dictionary to preload all data if enabled
        if self.preload_data:
            logger.info('Preloading enabled. Attempting to preload all data into memory.')
            mem_before = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
            logger.info(f'Memory before preloading: {mem_before:.2f} GB')
            for meta in self.file_metadata:
                file_path = meta['path']
                try:
                    self.data_cache[file_path] = torch.load(file_path, weights_only=False)
                    logger.info(f'Preloaded {os.path.basename(file_path)} into memory')
                except MemoryError as mem_err:
                    logger.warning(f'Memory error during preloading {file_path}: {mem_err}. Falling back to on-demand loading.')
                    self.data_cache = {}
                    self.preload_data = False  
                    break
                except Exception as e:
                    logger.error(f'Failed to preload {file_path}: {e}')
                mem_after = psutil.Process().memory_info().rss / (1024 ** 3)  
                logger.info(f'Memory after preloading: {mem_after:.2f} GB (Used: {mem_after - mem_before:.2f} GB)')
            else:
                # Summary log for data transfer
                if self.preload_data and self.data_cache:
                    total_files = len(self.data_cache)
                    logger.info(f'Data transfer summary: Successfully preloaded {total_files} files into memory.')
                else:
                    logger.info('Data transfer summary: On-demand mode enabled; files will be loaded and cached as needed during training.')
        else:
            logger.info('Preloading disabled. Using on-demand loading with LRU cache.')

        # Additional diagnostic log for preload status
        if self.preload_data:
            logger.info(f'Preload status: {len(self.data_cache)} files loaded into memory.')

        if max_samples is not None and max_samples > 0:
            accumulated = 0
            limited_metadata = []
            for meta in self.file_metadata:
                if accumulated >= max_samples:
                    break
                count = min(meta['count'], max_samples - accumulated)
                limited_metadata.append({'path': meta['path'], 'count': count})
                accumulated += count
            self.file_metadata = limited_metadata
            logger.info(f'Limited dataset to {accumulated} samples for faster training.')

    def _initialize_data(self):
        if self.file_prefix:
            logger.info(f"正在为前缀 '{self.file_prefix}' 加载指定的缓存文件... 检查目录: {self.processed_cache_dir}")
            glob_pattern = os.path.join(self.processed_cache_dir, f"{self.file_prefix}-*.pt")
            all_cache_files = glob.glob(glob_pattern)
            if not all_cache_files:
                logger.error(f"错误：在缓存目录 {self.processed_cache_dir} 中未找到任何匹配 '{self.file_prefix}-*.pt' 的文件。")
                return []
        else:
            logger.info(f"正在初始化完整数据集... 原始数据目录: {self.wiki_dir}, 缓存目录: {self.processed_cache_dir}")

            source_files = glob.glob(os.path.join(self.wiki_dir, "*.parquet"))
            if not source_files:
                logger.error(f"在 {self.wiki_dir} 中未找到任何 Parquet 文件。")
                return []
            
            logger.info(f"找到 {len(source_files)} 个原始 Parquet 文件。现在开始检查缓存状态...")

            files_to_process = []
            for file_path in source_files:
                cache_file_name = f"{Path(file_path).stem}.pt"
                meta_path = os.path.join(self.processed_cache_dir, cache_file_name + ".meta")
                if not os.path.exists(meta_path):
                    files_to_process.append(file_path)

            if files_to_process:
                logger.info(f"发现 {len(files_to_process)} 个文件需要处理和缓存。")
                for file_path in tqdm(files_to_process, desc="正在生成数据缓存"):
                    self._process_and_cache_file(file_path)
                logger.info("所有文件处理和缓存完成。")
            else:
                logger.info("所有原始文件均已缓存，无需重新生成。")
            all_cache_files = glob.glob(os.path.join(self.processed_cache_dir, "*.pt"))

        file_metadata = []
        for cache_path in all_cache_files:
            meta_path = cache_path + ".meta"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if 'count' in meta and meta['count'] > 0:
                        file_metadata.append({'path': cache_path, 'count': meta['count']})
                except Exception as e:
                    logger.warning(f"读取元数据文件 {meta_path} 失败: {e}，将忽略此缓存。")
        
        if not file_metadata:
            logger.error(f"未能从缓存目录加载任何有效的元数据 (模式: {self.file_prefix or '所有文件'})。")
            return []

        logger.info(f"已成功为 '{self.file_prefix or '所有文件'}' 加载 {len(file_metadata)} 个文件的元数据。")
        return file_metadata

    def _process_and_cache_file(self, file_path):
        cache_file_name = f"{Path(file_path).stem}.pt"
        cache_path = os.path.join(self.processed_cache_dir, cache_file_name)
        meta_path = cache_path + ".meta"

        logger.info(f"Processing source file: {os.path.basename(file_path)} -> Caching to: {os.path.basename(cache_path)}")
        
        processed_samples = []
        try:
            df = pd.read_parquet(file_path)
            
            if 'text' not in df.columns:
                logger.error(f"Column 'text' not found in {file_path}. Skipping.")
                return

            for text in tqdm(df['text'], desc=f"Tokenizing {os.path.basename(file_path)}", leave=False):
                if not isinstance(text, str):
                    continue

            
                word_count = len(text.split())
                if not (self.min_words <= word_count <= self.max_words):
                    continue
                
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                labels = tokens['input_ids'].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                processed_samples.append({
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'labels': labels.squeeze(0)
                })

            if processed_samples:
                torch.save(processed_samples, cache_path)
                with open(meta_path, 'w') as f:
                    json.dump({'count': len(processed_samples)}, f)
                logger.info(f"Successfully cached {len(processed_samples)} samples to {os.path.basename(cache_path)}")
            else:
                logger.warning(f"File {os.path.basename(file_path)} produced no valid samples within the word count range.")

        except Exception as e:
            logger.error(f"A critical error occurred while processing {file_path}: {e}", exc_info=True)

    def __len__(self):
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) > 0 else 0

    @lru_cache(maxsize=32)
    def _load_file_from_cache(self, file_path):
        short_name = os.path.basename(file_path)
        pid = os.getpid()
        # logger.info(f"[{pid}] 数据加载进程开始加载: {short_name}") # Verbose
        start_time = time.time()
        # Log memory before loading
        mem_before = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
        try:
            data = torch.load(file_path, weights_only=False)
            load_time = time.time() - start_time
            mem_after = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
            return data
        except Exception as e:
            logger.error(f"[{pid}] 加载缓存文件 {short_name} 失败: {e}", exc_info=True)
            return []

    def __getitem__(self, idx):
        try:
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")

            # 使用numpy的searchsorted快速查找索引所属的文件
            file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
            
            metadata = self.file_metadata[file_idx]
            file_path = metadata['path']
            
            # 计算在文件内的本地索引
            local_idx = idx - self.cumulative_sizes[file_idx]

            # 从预加载缓存或LRU缓存/磁盘加载文件数据
            if self.preload_data and file_path in self.data_cache:
                data = self.data_cache[file_path]
            else:
                data = self._load_file_from_cache(file_path)
            
            # 检查加载是否成功或索引是否越界
            if not data or local_idx >= len(data):
                raise ValueError(f"Data loading failed for index {idx} in file {os.path.basename(file_path)}")
                
            sample = data[int(local_idx)]

            if not isinstance(sample, dict) or 'input_ids' not in sample:
                 raise TypeError(f"Data item at index {idx} is not a valid dictionary, but {type(sample)}")

            return sample

        except Exception as e:
            error_msg = f"无法加载索引 {idx} 的数据: {e}"
            logger.warning(error_msg + "。将返回一个空的错误样本。")
            # 在出错时返回一个结构一致的空样本
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
            return {
                'input_ids': torch.full((self.max_length,), pad_token_id, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                'error': True # 标记为错误样本
            }

def prepare_wikipedia_data(
    wiki_dir,
    tokenizer,
    max_length=512,
    min_words=100,
    max_words=1000,
    batch_size=32,
    eval_batch_size=32,  
    num_workers=0,
    seed=42,
    languages=None,
    preload_data=False,
    max_samples=None
):
    cache_dir = os.path.join(os.getcwd(), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 分别为训练集和验证集创建独立的数据集实例
    logger.info("正在加载训练数据集 (train-*.pt)...")
    train_dataset = WikipediaDataset(
        wiki_dir=wiki_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        min_words=min_words,
        max_words=max_words,
        cache_dir=cache_dir,
        seed=seed,
        languages=languages,
        preload_data=preload_data,
        max_samples=max_samples,
        file_prefix="train"  # 指定只加载训练文件
    )

    logger.info("正在加载验证数据集 (validation-*.pt)...")
    val_dataset = WikipediaDataset(
        wiki_dir=wiki_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        min_words=min_words,
        max_words=max_words,
        cache_dir=cache_dir,
        seed=seed,
        languages=languages,
        preload_data=preload_data,
        max_samples=max_samples,
        file_prefix="validation"  # 指定只加载验证文件
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 计算验证集的批次数量
    val_batch_count = len(val_dataset) // eval_batch_size
    logger.info(f"验证批次大小: {eval_batch_size}, 预计验证批次数量: {val_batch_count}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def custom_collate_fn(batch):
    if isinstance(batch, dict):
        batch = [batch]
    
    # 检查batch是否为列表
    if not isinstance(batch, list):
        return None
    
    # 过滤掉None项
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 检查batch中的每个项
    valid_batch = []
    for i, item in enumerate(batch):
        if not isinstance(item, dict):
            continue
            
        if item.get('error', False):
            continue
            
        valid_batch.append(item)
    if not valid_batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(valid_batch)
    except Exception as e:
        return None

def log_expert_usage(model, expert_counts, total_tokens_for_stats, quiet):
    if quiet or total_tokens_for_stats == 0:
        return
        
    logger.info("--- 专家使用率分析 ---")
    
    # 计算专家使用百分比
    expert_percentages = (expert_counts.float() / total_tokens_for_stats * 100).cpu().numpy()
    
    # 获取路由器的专家使用EMA统计
    router_usage_stats = model.router.expert_usage_counts.cpu().float().numpy() * 100
    
    for i, percentage in enumerate(expert_percentages):
        logger.info(f"专家 {i}: 近期平均使用率 {percentage:.2f}%, EMA使用率 {router_usage_stats[i]:.2f}%")
    
    usage_std = np.std(expert_percentages)
    logger.info(f"专家使用率标准差: {usage_std:.2f}% (越低表示负载越均衡)")
    
    min_usage_threshold = 100.0 / model.num_experts * 0.5
    low_usage_experts = [i for i, p in enumerate(expert_percentages) if p < min_usage_threshold]
    if low_usage_experts:
        logger.warning(f"专家 {low_usage_experts} 使用率过低，可能需要调整负载均衡参数")
    
    logger.info("--- 分析结束 ---")

def train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, training_state, args, log_interval=1000, eval_interval=1000, quiet=False, val_loader=None, torch_dtype=torch.bfloat16):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    interval_total_loss = 0.0
    interval_ce_loss = 0.0
    interval_bal_loss = 0.0
    
    # For expert usage stats
    accumulated_expert_counts = torch.zeros(model.num_experts, device=device)
    accumulated_tokens_for_stats = 0
    
    # Calculate expert evaluation interval in batches
    expert_eval_interval_batches = max(1, 10000 // args.batch_size)

    for batch_idx, batch in enumerate(train_dataloader):
        if batch is None:
            logger.warning(f"跳过一个无效批次 (batch_idx: {batch_idx})")
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        with amp.autocast(device_type=device.type, dtype=torch_dtype):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
        with torch.no_grad():
            routing_indices = outputs["routing_indices"]
            accumulated_expert_counts += torch.bincount(routing_indices.view(-1), minlength=model.num_experts)
            accumulated_tokens_for_stats += routing_indices.numel()
        unscaled_loss = loss.detach().item()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        # BF16不需要梯度缩放
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in model.parameters() if p.grad is not None and p.requires_grad]), 2.0)
            
            # Only log a warning if the gradient norm is excessively high
            if grad_norm > args.grad_clip_norm * 2:
                if not quiet:
                    logger.warning(f'High gradient norm detected: {grad_norm.item():.4f}, clipping to {args.grad_clip_norm}')
            
            # 每500批次打印平均梯度范数（监控用途）
            if batch_idx % 500 == 0 and batch_idx > 0:
                avg_grad_norm = grad_norm.item()  # 这里简单用总norm作为平均的代理；如需精确，可除以参数组数
                logger.info(f'Batch {batch_idx}: Average gradient norm before clipping: {avg_grad_norm:.4f}')#batch_count 被重复增加了是正确的
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 
                max_norm=args.grad_clip_norm
            )
            
            # 直接更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 在梯度累积完成后调用scheduler
            if scheduler is not None:
                scheduler.step()
        
        # 累加未缩放的损失，以便正确计算平均损失
        total_loss += unscaled_loss
        
        interval_total_loss += unscaled_loss
        interval_ce_loss += outputs.get("ce_loss").item() if outputs.get("ce_loss") is not None else 0.0
        interval_bal_loss += outputs.get("load_balancing_loss").item() if outputs.get("load_balancing_loss") is not None else 0.0
        
        if (batch_idx + 1) % log_interval == 0 and not quiet:
            elapsed = time.time() - start_time
            
            # 计算并记录区间的平均损失
            avg_loss = interval_total_loss / log_interval
            avg_ce_loss = interval_ce_loss / log_interval
            avg_bal_loss = interval_bal_loss / log_interval
            
            avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float('inf')
            
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx + 1}/{len(train_dataloader)} | '
                        f'Avg Loss (last {log_interval}): {avg_loss:.4f} (CE: {avg_ce_loss:.4f} | Bal: {avg_bal_loss:.4f}) | '
                        f'Avg Perplexity: {avg_perplexity:.2f} | Time: {elapsed:.2f}s')

            # 为下一个区间重置
            start_time = time.time()
            interval_total_loss = 0.0
            interval_ce_loss = 0.0
            interval_bal_loss = 0.0
            save_interval_seconds = 7200 # 2 hours
            current_time = time.time()
            if current_time - training_state.last_save_time >= save_interval_seconds:
                logger.info(f"已达到 {save_interval_seconds / 3600:.0f} 小时保存间隔，正在保存可覆盖的定时检查点...")
                
                # 创建一个固定的、可覆盖的定时检查点文件名
                timed_checkpoint_path = os.path.join(args.save_dir, 'timed_checkpoint.pt')
                
                # 调用辅助函数保存完整检查点
                save_full_checkpoint(model, optimizer, scheduler, training_state, timed_checkpoint_path, logger)
                
                # 重置计时器
                training_state.last_save_time = current_time
        
        # --- 专家使用率统计 ---
        if (batch_idx + 1) % expert_eval_interval_batches == 0:
            log_expert_usage(model, accumulated_expert_counts, accumulated_tokens_for_stats, quiet)
            accumulated_expert_counts.zero_()
            accumulated_tokens_for_stats = 0

        if batch_idx > 0 and batch_idx % eval_interval == 0:
            if not quiet:
                logger.info(f"在第 {epoch} 轮第 {batch_idx} 批次进行评估")
            val_loss, val_perplexity = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_batches=args.eval_interval,
                quiet=quiet,
                torch_dtype=torch_dtype
            )
            model.train()  # 切回训练模式
            if not quiet:
                logger.info(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_perplexity:.4f}")
    
    return total_loss / len(train_dataloader)


def evaluate(model, dataloader, device, num_batches=None, quiet=False, torch_dtype=torch.bfloat16):
    model.eval()
    total_ce_loss = 0.0
    batch_count = 1
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False, disable=quiet)):
            if num_batches is not None and batch_idx >= num_batches:
                break
            if batch is None:
                continue
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            labels = batch['labels'].to(device)
            with amp.autocast(device_type=device.type, dtype=torch_dtype):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                if outputs["ce_loss"] is not None:
                    total_ce_loss += outputs["ce_loss"].item()
                    batch_count += 1
    if batch_count == 1:
        logger.warning("No batches processed during evaluation!")
        return float('inf'), float('inf')
        
    avg_loss = total_ce_loss / batch_count
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    if not quiet:
        logger.info(f"Evaluation: processed {batch_count} batches")
        logger.info(f"Avg loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
    return avg_loss, perplexity

def validate_expert_dimensions(expert_configs, load_models=False):
    logger.info("验证专家模型嵌入维度...")
    cache_file = "expert_dimensions_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # 检查缓存中的模型路径是否与当前配置匹配
            cached_paths = cache_data.get('model_paths', [])
            current_paths = [config['path'] for config in expert_configs]
            
            if set(cached_paths) == set(current_paths):
                logger.info(f"使用缓存的嵌入维度: {cache_data['embed_dim']}")
                if not load_models:
                    return cache_data['embed_dim'], None
        except Exception as e:
            logger.warning(f"读取缓存文件失败: {str(e)}")
    
    reference_expert = ExpertModel(
        model_path=expert_configs[0]['path'],
        device_map=expert_configs[0].get('device_map', 'auto') if load_models else 'cpu',
        load_in_4bit=expert_configs[0].get('load_in_4bit', False),
        load_in_8bit=expert_configs[0].get('load_in_8bit', False),
        disable_marlin=True, # 禁用Marlin以支持训练
        tokenizer_path=expert_configs[0].get('tokenizer_path', "/mnt/data/zhangjinhao/Llama-3.2-3B-Instruct")
    )
    
    logger.info(f"加载参考专家模型: {expert_configs[0]['path']} 到设备 {expert_configs[0].get('device_map', 'auto')}")
    success = reference_expert.load()
    if not success:
        raise RuntimeError("无法加载参考专家模型")
    
    reference_dim = reference_expert.get_embedding_dim()
    logger.info(f"参考嵌入维度: {reference_dim}")
    
    # 如果不需要加载模型，释放参考模型内存
    if not load_models:
        loaded_experts = []
        del reference_expert
        torch.cuda.empty_cache()
    else:
        loaded_experts = [reference_expert]
    
    # 检查其他专家模型
    for i, config in enumerate(expert_configs[1:], 1):
        if load_models:
            device_map = config.get('device_map', 'auto')
            logger.info(f"加载专家模型 {i}: {config['path']} 到设备 {device_map}")
            
            expert = ExpertModel(
                model_path=config['path'],
                device_map=device_map,
                load_in_4bit=config.get('load_in_4bit', False),
                load_in_8bit=config.get('load_in_8bit', False),
                disable_marlin=True, # 禁用Marlin以支持训练
                tokenizer_path=config.get('tokenizer_path', "/mnt/data/zhangjinhao/Llama-3.2-3B-Instruct")
            )
        else:
            # 如果只是验证维度，使用CPU加载以节省内存
            expert = ExpertModel(
                model_path=config['path'],
                device_map='cpu'
            )
            
        success = expert.load()
        if not success:
            raise RuntimeError(f"无法加载专家模型 {i}")
        
        dim = expert.get_embedding_dim()
        logger.info(f"专家 {i} 嵌入维度: {dim}")
        
        if dim != reference_dim:
            logger.warning(f"专家 {i} 的嵌入维度 ({dim}) 与参考维度 ({reference_dim}) 不匹配")
            raise ValueError(f"所有专家模型必须有相同的嵌入维度，但专家 {i} 的维度为 {dim}，而参考维度为 {reference_dim}")
        
        # 如果不需要加载模型，释放内存
        if not load_models:
            del expert
            torch.cuda.empty_cache()
        else:
            loaded_experts.append(expert)
    
    logger.info("所有专家模型的嵌入维度验证通过")
    
    # 保存缓存
    try:
        cache_data = {
            'embed_dim': reference_dim,
            'model_paths': [config['path'] for config in expert_configs],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.debug(f"已保存嵌入维度缓存到 {cache_file}")
    except Exception as e:
        logger.warning(f"保存缓存文件失败: {str(e)}")
    
    return reference_dim, loaded_experts if load_models else None

def load_model_with_mismatch(model, state_dict, logger):
    # 获取当前模型的参数字典
    model_state = model.state_dict()
    
    new_state_dict = {}
    missing_keys = []
    
    # 遍历保存在检查点里的每一个参数
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            # 只有名字和形状都匹配，才将其加入到新的参数字典中
            new_state_dict[k] = v
        else:
            # 否则，记录下来这个不匹配的参数
            missing_keys.append(k)
    
    model.load_state_dict(new_state_dict, strict=False)
    
    logger.info(f"已加载 {len(new_state_dict)}/{len(state_dict)} 个参数")
    logger.info(f"忽略了 {len(missing_keys)} 个不匹配的参数")

def train(args):
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    torch_dtype = None # 让 from_pretrained 自动推断
    logger.info(f"将使用模型默认精度进行加载")

    # 清理内存
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        gc.collect()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载专家模型配置
    expert_configs = []
    embedding_source_path = "/mnt/data/zhangjinhao/Llama-3.2-3B-Instruct"  # 非量化模型路径，仅用于提取embedding层
    
    if args.expert_paths:
        logger.info(f"从命令行参数加载专家模型路径: {args.expert_paths}")
        for path in args.expert_paths:
            # 对于GGUF或预量化模型，我们不在此处设置量化标志
            # ExpertModel类将处理加载
            expert_configs.append({'path': path})
    else:
        logger.info("未通过命令行指定专家路径，使用默认的GGUF量化模型列表。")
        default_expert_paths = [
            'path/to/expert1',
            'path/to/expert2',
            'path/to/expert3',
            'path/to/expert4'
        ]
        for path in default_expert_paths:
            expert_configs.append({'path': path})
    
    logger.info("开始智能分配GPU资源...")
    expert_configs = allocate_models_to_gpus(expert_configs, gpu_threshold_gb=10.0)
    
    logger.info("=" * 60)
    logger.info("步骤 1: 加载专家模型")
    try:
        embed_dim, loaded_experts = validate_expert_dimensions(expert_configs, load_models=True)
        if loaded_experts is None:
            logger.critical("未能加载任何专家模型，测试无法继续。")
            exit(1)
        logger.info(f"成功加载 {len(loaded_experts)} 个专家模型")
    except Exception as e:
        logger.critical(f"加载或验证专家模型时发生致命错误: {e}")
        logger.debug(traceback.format_exc())
        exit(1)
    
    # 获取主设备，用于主干网络和路由器
    main_device = device
    if torch.cuda.device_count() > 1:
        # 如果有多个GPU，使用第一个GPU作为主设备
        main_device = torch.device('cuda:0')
        logger.info(f"使用 {main_device} 作为主设备用于backbone和router")
    
    logger.info("步骤 2: 初始化MoE模型并加载非量化模型提取embedding层")
    
    # 准备路由器配置
    router_config = {
        'num_heads': args.router_heads,
        'transformer_layers': args.transformer_layers,
        'dynamic_k': args.dynamic_routing,
        'min_k': args.min_k,
        'complexity_factor': args.complexity_factor,
        'dropout': args.router_dropout,
        'gating_temperature': args.gating_temperature
    }
    
    # 如果启用动态路由，记录相关信息
    if args.dynamic_routing:
        logger.info(f"已启用动态路由 - 最小k值: {args.min_k}, 最大k值: {args.top_k}, 复杂度因子: {args.complexity_factor}")
    
    model = MoEModel(
        expert_configs, 
        top_k=args.top_k, 
        device=main_device, 
        loaded_experts=loaded_experts, 
        current_epoch=0,
        embedding_source_path=embedding_source_path,
        router_config=router_config,
        torch_dtype=torch.float16, # 保持一个默认值，但加载时会优先使用模型自身的dtype
        load_balance_loss_coef=args.load_balance_loss_coef,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    logger.info("步骤 3: 初始化优化器和学习率调度器")
    optimizer = None
    scheduler = None
    train_loader, val_loader = None, None
    current_max_samples = -1 
    training_state = TrainingState()
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else os.path.join(args.save_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path) and not args.from_scratch:
        logger.info(f"找到检查点文件: {checkpoint_path}")
        logger.info(f"继续训练模式: 从检查点加载模型状态")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        training_state.load_state(checkpoint.get('training_state', {}))
        
        # 修改加载方式，只加载匹配的参数，忽略不匹配的参数
        if 'model_state_dict' in checkpoint:
            load_model_with_mismatch(model, checkpoint['model_state_dict'], logger)
        
        logger.info(f"检查点恢复：当前epoch为 {training_state.current_epoch}。重新构建优化器状态...")
        
        # 移除了微调逻辑，总是创建只包含路由器和词嵌入的优化器
        logger.info("从路由学习阶段恢复。创建包含2个参数组的优化器。")
        model._freeze_experts(fine_tune_layers=0)
        param_groups = [
            {'params': model.router.parameters(), 'lr': args.learning_rate},
            {'params': model.shared_embed.parameters(), 'lr': args.embedding_lr}
        ]

        # 创建优化器
        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                logger.info("从检查点恢复，尝试使用 8-bit AdamW 优化器。")
                optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=args.weight_decay)
            except ImportError:
                logger.warning("bitsandbytes 未安装或无法加载，将回退到标准的 AdamW 优化器。")
                optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        # 加载优化器和调度器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 调度器状态加载将在训练循环内处理
        logger.info(f"已加载检查点，继续从轮次 {training_state.current_epoch} 训练")
    else:
        if args.from_scratch:
            logger.info("从头训练模式: 创建初始优化器")
        else:
            if args.checkpoint_path:
                logger.warning(f"指定的检查点文件不存在: {args.checkpoint_path}")
            logger.info("未找到检查点文件，将从头开始训练并创建初始优化器")

        model._freeze_experts(fine_tune_layers=0)
    
        # 为不同部分设置不同学习率
        param_groups = [
            {'params': model.router.parameters(), 'lr': args.learning_rate},
            {'params': model.shared_embed.parameters(), 'lr': args.embedding_lr}
        ]

        logger.info(f"配置初始优化器 - Router LR: {args.learning_rate}, Embedding LR: {args.embedding_lr}")

        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                logger.info("正在使用 8-bit AdamW 优化器以节省显存。")
                optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=args.weight_decay)
            except ImportError:
                logger.warning("bitsandbytes 未安装，将回退到标准的 AdamW 优化器。")
                optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        else:
            logger.info("正在使用标准的 32-bit AdamW 优化器。")
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    logger.info("Optimizer initialized")
    
    # 训练循环
    for epoch in range(training_state.current_epoch, args.epochs):
        if epoch + 1 < args.switch_epoch:
            effective_max_samples = args.initial_samples
        else:
            effective_max_samples = args.later_samples

        # 检查数据加载器是否需要更新
        if effective_max_samples != current_max_samples:
            logger.info(f"样本数量变化或首次加载 (epoch {epoch+1})，将创建新的数据加载器 (max_samples: {effective_max_samples})")
            current_max_samples = effective_max_samples
            
            train_loader, val_loader = prepare_wikipedia_data(
                wiki_dir=args.data_dir,
                tokenizer=model.tokenizer,  
                max_length=args.max_length,
                min_words=args.min_words,
                max_words=args.max_words,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                languages=args.languages,
                preload_data=args.preload_data,
                max_samples=effective_max_samples
            )
            
            # Recalculate total steps for the scheduler for the rest of the training
            remaining_epochs = args.epochs - epoch
            total_steps = len(train_loader) * remaining_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=args.warmup_steps, 
                num_training_steps=total_steps,
                last_epoch=-1  # Reset if needed
            )
            # 如果从检查点恢复，也加载调度器状态
            if 'scheduler_state_dict' in locals().get('checkpoint', {}):
                 try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 except Exception as e:
                    logger.warning(f"加载学习率调度器状态失败: {str(e)}")

        else:
            logger.info(f'Epoch {epoch+1}: 使用与上一轮相同的样本数量 (max_samples: {current_max_samples})，跳过数据加载。')
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        model.router.current_epoch = epoch
        
        avg_train_loss = train_epoch(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            training_state=training_state,
            args=args,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            quiet=args.quiet,
            val_loader=val_loader,
            torch_dtype=torch.float16 # 保持一个默认值
        )
        
        # 验证阶段
        avg_val_loss, val_perplexity = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            num_batches=args.eval_interval,
            quiet=args.quiet,
            torch_dtype=torch.float16 # 保持一个默认值
        )
        
        if not args.quiet:
            logger.info(f"验证损失: {avg_val_loss:.4f}, 验证困惑度: {val_perplexity:.4f}")
        
        # 更新学习率 - a step is already called in train_epoch
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            if not args.quiet:
                logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 更新训练状态
        training_state.update(epoch + 1, avg_train_loss, avg_val_loss, val_perplexity)
        
        # 更新路由器的温度参数
        if hasattr(model.router, 'gating_temperature') and args.anneal_temperature:
            progress = min(1.0, (epoch + 1) / args.epochs)
            new_temp = args.initial_temperature * (args.final_temperature / args.initial_temperature) ** progress
            model.router.gating_temperature = new_temp
            if not args.quiet and epoch % 1 == 0:
                logger.info(f"路由温度参数更新为: {new_temp:.4f}")
        if not args.quiet:
            logger.info("开始在轮次结束后保存模型和检查点...")

        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        save_full_checkpoint(model, optimizer, scheduler, training_state, checkpoint_path, logger)

        if training_state.is_best_model():
            if not args.quiet:
                logger.info(f"新最佳模型! 困惑度: {val_perplexity:.4f}. 保存最佳模型的可训练参数...")
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            temp_best_model_path = best_model_path + ".tmp"
            try:
                trainable_params = extract_unfrozen_parameters(model)
                torch.save(trainable_params, temp_best_model_path)
                os.rename(temp_best_model_path, best_model_path)
                if not args.quiet:
                    logger.info(f"最佳模型的可训练参数已保存到: {best_model_path}")
            except Exception as e:
                logger.error(f"保存最佳模型到 {best_model_path} 失败: {e}")
                if os.path.exists(temp_best_model_path):
                    os.remove(temp_best_model_path)
    
    logger.info("训练完成!")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger.info("将多进程启动方法设置为 'spawn' 以提高稳定性。")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="训练混合专家模型")
    # 核心参数
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--eval", action="store_true", help="评估模型")
    parser.add_argument("--from_scratch", action="store_true", help="从头开始训练，忽略现有检查点")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="指定要加载的检查点文件路径")
    parser.add_argument("--checkpoint_prefix", type=str, default="checkpoint", help="检查点文件名前缀")
    parser.add_argument("--save_interval", type=int, default=1, help="保存检查点的轮次间隔，设为1表示每轮都保存")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小 ")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="评估/推理时的批次大小，默认为32（增加批次大小以减少batch_count）")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率 (微调时建议使用较小的值，如1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="优化器中的权重衰减 (L2正则化)")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/zhangjinhao/data", help="数据目录")
    parser.add_argument("--max_length", type=int, default=512, help="序列长度")
    parser.add_argument("--save_dir", type=str, default="/mnt/data/zhangjinhao/moe_output", help="保存目录")
    parser.add_argument("--expert_paths", nargs="+", help="专家模型路径")
    parser.add_argument("--top_k", type=int, default=3, help="专家路由器top_k值")
    parser.add_argument("--device", type=str, default=None, help="指定设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--low_memory", action="store_true", help="启用低内存模式")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="预热步数")
    parser.add_argument("--log_interval", type=int, default=1000, help="日志记录间隔")
    parser.add_argument("--eval_interval", type=int, default=4000, help="评估间隔")  
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试数据比例")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数 (默认0，避免多进程内存问题)")
    parser.add_argument("--min_words", type=int, default=100, help="最小单词数")
    parser.add_argument("--max_words", type=int, default=1000, help="最大单词数")
    parser.add_argument("--languages", nargs="+", help="要使用的语言列表，例如 en zh fr")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--load_balance_alpha", type=float, default=0.9, 
                        help="专家使用率指数移动平均的衰减率，值越大表示历史使用率影响越大")
    parser.add_argument("--load_balance_strength", type=float, default=1.5, 
                        help="负载均衡强度系数，控制偏置调整的幅度，0表示不调整")
    parser.add_argument("--load_balance_loss_coef", type=float, default=0.02, 
                        help="负载均衡损失的权重系数 (alpha)")
    parser.add_argument("--embedding_lr", type=float, default=1e-6,
                        help="专门为词嵌入层设置的较小学习率")
    parser.add_argument("--preload_data", action="store_true", default=False, help="是否预加载所有数据到内存以加速训练 (默认: False)")
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=True, help="使用8位优化器以减少内存消耗 (默认: True)")
    parser.add_argument("--initial_samples", type=int, default=70000, help="Samples for first 1-2 epochs")
    parser.add_argument("--later_samples", type=int, default=300000, help="Samples for subsequent epochs")
    parser.add_argument("--switch_epoch", type=int, default=3, help="Epoch to switch to later_samples")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="梯度累积步数，用于模拟更大批次而不增加内存")
    parser.add_argument("--router_heads", type=int, default=4, help="路由器中注意力机制的头数")
    parser.add_argument("--transformer_layers", type=int, default=2, help="路由器中Transformer编码器的层数")
    parser.add_argument("--dynamic_routing", action="store_true", default=True, help="启用动态路由阈值，根据输入复杂度调整top-k值")
    parser.add_argument("--min_k", type=int, default=1, help="动态路由时的最小k值")
    parser.add_argument("--complexity_factor", type=float, default=0.7, help="复杂度因子，控制动态k值的计算")
    parser.add_argument("--router_dropout", type=float, default=0.1, help="路由器中的dropout比例")
    parser.add_argument("--gating_temperature", type=float, default=1.0, help="路由软化温度参数，控制路由的软硬程度")
    parser.add_argument("--anneal_temperature", action="store_true", default=True, help="是否在训练过程中逐渐降低温度参数")
    parser.add_argument("--initial_temperature", type=float, default=1.2, help="初始路由温度")
    parser.add_argument("--final_temperature", type=float, default=0.5, help="最终路由温度")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16"], help="训练精度，'auto'表示由模型自身决定")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点以减少内存使用 (默认开启以节省内存)")
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    args = parser.parse_args()
    
    # 设置内存优化
    if args.low_memory:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("已启用低内存模式")
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    if not args.debug:
        logging.getLogger("transformers").setLevel(logging.ERROR)

    # 默认使用训练模式
    if not args.train and not args.eval:
        args.train = True
    
    if args.train:
        train(args)
    
    if args.eval:
        pass
