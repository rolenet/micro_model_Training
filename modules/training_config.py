import os
import json
from typing import Dict, List, Optional, Union

class TrainingConfig:
    """微调配置模块，用于设置微调参数"""
    
    def __init__(self, config_path: str = None):
        # 修改配置文件路径，使用项目根目录
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default_config.json")
        self.config = self._load_config(config_path)
        self.training_params = self.config.get("training", {})
        self.system_params = self.config.get("system", {})
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "training": {
                    "batch_size": 4,
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "max_steps": 1000,
                    "gradient_accumulation_steps": 1,
                    "warmup_ratio": 0.03,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "max_seq_length": 512,
                    "weight_decay": 0.01,
                    "optimizer": "adamw_torch",
                    "scheduler": "cosine"
                },
                "system": {
                    "save_steps": 100,
                    "eval_steps": 100,
                    "logging_steps": 10,
                    "output_dir": "./outputs",
                    "log_dir": "./logs"
                }
            }
    
    def get_training_params(self) -> Dict:
        """获取训练参数"""
        return self.training_params
    
    def get_system_params(self) -> Dict:
        """获取系统参数"""
        return self.system_params
    
    def update_training_params(self, params: Dict) -> None:
        """更新训练参数"""
        self.training_params.update(params)
    
    def update_system_params(self, params: Dict) -> None:
        """更新系统参数"""
        self.system_params.update(params)
    
    def save_config(self, config_path: str = None) -> bool:
        """保存配置到文件"""
        if config_path is None:
            config_path = "../config/default_config.json"
        
        try:
            # 更新配置
            self.config["training"] = self.training_params
            self.config["system"] = self.system_params
            
            # 确保目录存在
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # 保存到文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def get_lora_config(self) -> Dict:
        """获取LoRA配置"""
        return {
            "r": self.training_params.get("lora_r", 8),
            "lora_alpha": self.training_params.get("lora_alpha", 16),
            "lora_dropout": self.training_params.get("lora_dropout", 0.05),
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    
    def get_training_arguments(self) -> Dict:
        """获取训练参数，用于传递给Trainer"""
        output_dir = os.path.abspath(self.system_params.get("output_dir", "./outputs"))
        return {
            "output_dir": output_dir,
            "num_train_epochs": self.training_params.get("num_epochs", 3),
            "per_device_train_batch_size": self.training_params.get("batch_size", 4),
            "gradient_accumulation_steps": self.training_params.get("gradient_accumulation_steps", 1),
            "learning_rate": self.training_params.get("learning_rate", 2e-4),
            "max_steps": self.training_params.get("max_steps", 1000),
            "warmup_ratio": self.training_params.get("warmup_ratio", 0.03),
            "logging_steps": self.system_params.get("logging_steps", 10),
            "save_steps": self.system_params.get("save_steps", 100),
            "eval_steps": self.system_params.get("eval_steps", 100),
            "weight_decay": self.training_params.get("weight_decay", 0.01),
            "optim": self.training_params.get("optimizer", "adamw_torch"),
            "lr_scheduler_type": self.training_params.get("scheduler", "cosine"),
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "report_to": "tensorboard",
        }


if __name__ == "__main__":
    # 测试配置功能
    config = TrainingConfig()
    print(f"训练参数: {config.get_training_params()}")
    print(f"系统参数: {config.get_system_params()}")
    
    # 更新参数
    config.update_training_params({"batch_size": 8, "learning_rate": 1e-4})
    print(f"更新后的训练参数: {config.get_training_params()}")
    
    # 获取LoRA配置
    print(f"LoRA配置: {config.get_lora_config()}")
    
    # 获取训练参数
    print(f"训练参数: {config.get_training_arguments()}")