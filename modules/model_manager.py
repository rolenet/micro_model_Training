import os
import json
from typing import Dict, List, Optional, Union
from huggingface_hub import snapshot_download, list_models, login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

class ModelManager:
    """模型管理模块，用于下载和管理模型"""
    
    def __init__(self, config_path: str = None):
        # 修改配置文件路径，使用项目根目录
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default_config.json")
        self.config = self._load_config(config_path)
        self.models_dir = os.path.abspath(self.config["model"]["local_models_dir"])
        os.makedirs(self.models_dir, exist_ok=True)
        self.current_model = None
        self.current_tokenizer = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "model": {
                    "default_model": "meta-llama/Llama-2-7b-hf",
                    "local_models_dir": "./models"
                }
            }
    
    def list_local_models(self) -> List[str]:
        """列出本地已下载的模型"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path) and self._is_valid_model_dir(item_path):
                models.append(item)
        
        return models
    
    def _is_valid_model_dir(self, directory: str) -> bool:
        """检查目录是否为有效的模型目录"""
        # 检查是否包含模型配置文件
        config_file = os.path.join(directory, "config.json")
        return os.path.exists(config_file)
    
    def search_huggingface_models(self, query: str = "", model_type: str = "causal-lm", limit: int = 20) -> List[Dict]:
        """搜索HuggingFace上的模型"""
        try:
            models = list_models(filter=model_type, search=query, limit=limit)
            return [{
                "id": model.modelId,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag
            } for model in models]
        except Exception as e:
            print(f"搜索模型失败: {e}")
            return []
    
    def download_model(self, model_id: str, use_auth_token: Optional[str] = None) -> str:
        """从HuggingFace下载模型"""
        try:
            # 如果提供了token，先登录
            if use_auth_token:
                login(token=use_auth_token)
            
            # 下载模型到本地
            model_dir = os.path.join(self.models_dir, model_id.split("/")[-1])
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                token=use_auth_token,
                ignore_patterns=["*.safetensors", "*.bin", "*.pt"] if "*-gguf" in model_id else None
            )
            
            return model_dir
        except Exception as e:
            print(f"下载模型失败: {e}")
            return ""
    
    def load_model(self, model_path: str, device_map: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False) -> bool:
        """加载模型"""
        try:
            # 检查路径是否为HuggingFace模型ID
            if not os.path.exists(model_path) and "/" in model_path:
                print(f"模型路径不存在，尝试从HuggingFace下载: {model_path}")
                model_path = self.download_model(model_path)
                if not model_path:
                    return False
            
            # 加载tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 根据量化选项加载模型
            if load_in_8bit:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    load_in_8bit=True,
                    torch_dtype=torch.float16
                )
            elif load_in_4bit:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    load_in_4bit=True,
                    torch_dtype=torch.float16
                )
            else:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            print(f"模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def load_model_with_unsloth(self, model_path: str, device_map: str = "auto", max_seq_length: int = 2048) -> bool:
        """使用unsloth加载模型（针对微调优化）"""
        try:
            from unsloth import FastLanguageModel
            
            # 检查路径是否为HuggingFace模型ID
            if not os.path.exists(model_path) and "/" in model_path:
                print(f"模型路径不存在，尝试从HuggingFace下载: {model_path}")
                model_path = self.download_model(model_path)
                if not model_path:
                    return False
            
            # 使用unsloth加载模型和tokenizer
            self.current_model, self.current_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
            )
            
            print(f"使用unsloth加载模型成功: {model_path}")
            return True
        except Exception as e:
            print(f"使用unsloth加载模型失败: {e}，尝试使用标准方法加载")
            return self.load_model(model_path, device_map)
    
    def get_model_info(self, model_path: str) -> Dict:
        """获取模型信息"""
        try:
            config = AutoConfig.from_pretrained(model_path)
            info = {
                "name": os.path.basename(model_path),
                "path": model_path,
                "architecture": config.architectures[0] if hasattr(config, "architectures") and config.architectures else "Unknown",
                "vocab_size": config.vocab_size if hasattr(config, "vocab_size") else "Unknown",
                "hidden_size": config.hidden_size if hasattr(config, "hidden_size") else "Unknown",
                "num_layers": config.num_hidden_layers if hasattr(config, "num_hidden_layers") else "Unknown",
                "num_attention_heads": config.num_attention_heads if hasattr(config, "num_attention_heads") else "Unknown",
                "max_position_embeddings": config.max_position_embeddings if hasattr(config, "max_position_embeddings") else "Unknown",
            }
            return info
        except Exception as e:
            print(f"获取模型信息失败: {e}")
            return {"name": os.path.basename(model_path), "path": model_path, "error": str(e)}


if __name__ == "__main__":
    # 测试模型管理功能
    manager = ModelManager()
    local_models = manager.list_local_models()
    print(f"本地模型: {local_models}")
    
    # 搜索HuggingFace模型
    hf_models = manager.search_huggingface_models(query="llama", limit=5)
    print(f"HuggingFace模型搜索结果: {hf_models}")