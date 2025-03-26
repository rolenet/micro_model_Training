import os
import json
import torch
from typing import Dict, List, Optional, Union, Callable
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from unsloth import FastLanguageModel
from modules.model_manager import ModelManager
from modules.dataset_manager import DatasetManager
from modules.training_config import TrainingConfig
from modules.training_monitor import TrainingMonitor

class ModelTrainer:
    """模型微调模块，用于实现模型微调的核心功能"""
    
    def __init__(self, model_manager: ModelManager, dataset_manager: DatasetManager, 
                 training_config: TrainingConfig, training_monitor: TrainingMonitor):
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.training_config = training_config
        self.training_monitor = training_monitor
        self.trainer = None
    
    def prepare_model_for_training(self) -> bool:
        """准备模型进行微调"""
        try:
            if self.model_manager.current_model is None:
                raise ValueError("模型未加载，请先加载模型")
            
            # 获取LoRA配置
            lora_config = LoraConfig(
                r=self.training_config.get_lora_config()["r"],
                lora_alpha=self.training_config.get_lora_config()["lora_alpha"],
                lora_dropout=self.training_config.get_lora_config()["lora_dropout"],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # 检查是否已经是Unsloth模型
            if hasattr(self.model_manager.current_model, "is_unsloth") and self.model_manager.current_model.is_unsloth:
                # 对于Unsloth模型，使用其内置的LoRA方法
                FastLanguageModel.get_peft_model(
                    self.model_manager.current_model,
                    lora_config,
                    train_on_inputs=True
                )
            else:
                # 对于普通模型，使用PEFT库的LoRA方法
                self.model_manager.current_model = get_peft_model(self.model_manager.current_model, lora_config)
            
            print("模型已准备好进行微调")
            return True
        except Exception as e:
            print(f"准备模型进行微调失败: {e}")
            return False
    
    def prepare_dataset_for_training(self) -> bool:
        """准备数据集进行微调"""
        try:
            if self.dataset_manager.current_dataset is None:
                raise ValueError("数据集未加载，请先加载数据集")
            
            # 获取训练参数
            training_params = self.training_config.get_training_params()
            text_column = training_params.get("text_column", "text")
            instruction_column = training_params.get("instruction_column", "instruction")
            response_column = training_params.get("response_column", "response")
            
            # 准备数据集
            self.train_dataset = self.dataset_manager.prepare_for_training(
                text_column=text_column,
                instruction_column=instruction_column,
                response_column=response_column
            )
            
            print("数据集已准备好进行微调")
            return True
        except Exception as e:
            print(f"准备数据集进行微调失败: {e}")
            return False
    
    def _format_instruction_dataset(self, examples):
        """格式化指令数据集"""
        instruction_column = self.training_config.get_training_params().get("instruction_column", "instruction")
        response_column = self.training_config.get_training_params().get("response_column", "response")
        
        instructions = examples[instruction_column]
        responses = examples[response_column]
        
        # 创建指令格式的文本
        texts = []
        for instruction, response in zip(instructions, responses):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            texts.append(text)
        
        # 使用tokenizer处理文本
        tokenizer = self.model_manager.current_tokenizer
        encodings = tokenizer(texts, truncation=True, padding="max_length", 
                             max_length=self.training_config.get_training_params().get("max_seq_length", 512))
        
        return encodings
    
    def _format_text_dataset(self, examples):
        """格式化文本数据集"""
        text_column = self.training_config.get_training_params().get("text_column", "text")
        
        # 使用tokenizer处理文本
        tokenizer = self.model_manager.current_tokenizer
        encodings = tokenizer(examples[text_column], truncation=True, padding="max_length", 
                             max_length=self.training_config.get_training_params().get("max_seq_length", 512))
        
        return encodings
    
    def setup_trainer(self) -> bool:
        """设置训练器"""
        try:
            # 准备模型和数据集
            if not self.prepare_model_for_training() or not self.prepare_dataset_for_training():
                return False
            
            # 获取训练参数
            training_args_dict = self.training_config.get_training_arguments()
            training_args = TrainingArguments(**training_args_dict)
            
            # 设置数据整理器
            tokenizer = self.model_manager.current_tokenizer
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            # 创建训练器
            self.trainer = Trainer(
                model=self.model_manager.current_model,
                args=training_args,
                train_dataset=self.train_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # 设置回调函数，用于监控训练过程
            self.trainer.add_callback(self._create_training_callback())
            
            print("训练器设置完成")
            return True
        except Exception as e:
            print(f"设置训练器失败: {e}")
            return False
    
    def _create_training_callback(self):
        """创建训练回调函数"""
        from transformers.trainer_callback import TrainerCallback
        
        monitor = self.training_monitor
        
        class TrainingMonitorCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                monitor.start_training()
            
            def on_train_end(self, args, state, control, **kwargs):
                monitor.end_training()
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step = state.global_step
                    loss = logs.get("loss", 0)
                    learning_rate = logs.get("learning_rate", 0)
                    
                    monitor.log_training_step(step, loss, learning_rate, logs)
                    monitor.update_ui({
                        "step": step,
                        "loss": loss,
                        "learning_rate": learning_rate,
                        "progress": step / state.max_steps if state.max_steps else 0
                    })
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    step = state.global_step
                    monitor.log_evaluation_step(step, metrics)
        
        return TrainingMonitorCallback()
    
    def start_training(self) -> bool:
        """开始训练"""
        try:
            if self.trainer is None:
                if not self.setup_trainer():
                    return False
            
            # 开始训练
            self.trainer.train()
            
            # 保存模型
            output_dir = self.training_config.get_system_params().get("output_dir", "./outputs")
            self.trainer.save_model(output_dir)
            self.model_manager.current_tokenizer.save_pretrained(output_dir)
            
            # 保存训练日志
            self.training_monitor.save_logs()
            
            print("训练完成，模型已保存")
            return True
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """使用微调后的模型生成文本"""
        try:
            if self.model_manager.current_model is None:
                raise ValueError("模型未加载，请先加载模型")
            
            # 准备输入
            tokenizer = self.model_manager.current_tokenizer
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model_manager.current_model.device) for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model_manager.current_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
        except Exception as e:
            print(f"生成文本失败: {e}")
            return ""


if __name__ == "__main__":
    # 测试训练功能
    from modules.model_manager import ModelManager
    from modules.dataset_manager import DatasetManager
    from modules.training_config import TrainingConfig
    from modules.training_monitor import TrainingMonitor
    
    model_manager = ModelManager()
    dataset_manager = DatasetManager()
    training_config = TrainingConfig()
    training_monitor = TrainingMonitor()
    
    # 加载模型和数据集
    model_manager.load_model_with_unsloth("meta-llama/Llama-2-7b-hf")
    dataset_manager.load_dataset("tatsu-lab/alpaca")
    
    # 创建训练器
    trainer = ModelTrainer(model_manager, dataset_manager, training_config, training_monitor)
    
    # 设置训练器
    if trainer.setup_trainer():
        # 开始训练
        trainer.start_training()