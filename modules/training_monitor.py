import os
import json
import time
from typing import Dict, List, Optional, Union, Callable
import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor:
    """训练监控模块，用于监控和展示微调过程"""
    
    def __init__(self, log_dir: str = None):
        # 修改日志目录路径，使用项目根目录下的logs文件夹
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.training_logs = []
        self.evaluation_logs = []
        self.start_time = None
        self.end_time = None
    
    def start_training(self) -> None:
        """开始训练，记录开始时间"""
        self.start_time = time.time()
        self.training_logs = []
        self.evaluation_logs = []
    
    def end_training(self) -> None:
        """结束训练，记录结束时间"""
        self.end_time = time.time()
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, additional_info: Dict = None) -> None:
        """记录训练步骤信息"""
        log_entry = {
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "time": time.time()
        }
        
        if additional_info:
            log_entry.update(additional_info)
        
        self.training_logs.append(log_entry)
    
    def log_evaluation_step(self, step: int, metrics: Dict) -> None:
        """记录评估步骤信息"""
        log_entry = {
            "step": step,
            "metrics": metrics,
            "time": time.time()
        }
        
        self.evaluation_logs.append(log_entry)
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要信息"""
        if not self.training_logs:
            return {"error": "没有训练日志"}
        
        # 计算训练时间
        training_time = 0
        if self.start_time:
            end_time = self.end_time if self.end_time else time.time()
            training_time = end_time - self.start_time
        
        # 获取损失曲线
        steps = [log["step"] for log in self.training_logs]
        losses = [log["loss"] for log in self.training_logs]
        
        # 计算平均损失和最终损失
        avg_loss = sum(losses) / len(losses) if losses else 0
        final_loss = losses[-1] if losses else 0
        
        # 获取评估指标
        eval_metrics = {}
        if self.evaluation_logs:
            last_eval = self.evaluation_logs[-1]
            eval_metrics = last_eval["metrics"]
        
        return {
            "total_steps": len(self.training_logs),
            "training_time": training_time,
            "avg_loss": avg_loss,
            "final_loss": final_loss,
            "eval_metrics": eval_metrics
        }
    
    def plot_training_loss(self, save_path: Optional[str] = None) -> str:
        """绘制训练损失曲线"""
        if not self.training_logs:
            return "没有训练日志，无法绘制损失曲线"
        
        steps = [log["step"] for log in self.training_logs]
        losses = [log["loss"] for log in self.training_logs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            return f"损失曲线已保存到: {save_path}"
        else:
            save_path = os.path.join(self.log_dir, f"loss_curve_{int(time.time())}.png")
            plt.savefig(save_path)
            return f"损失曲线已保存到: {save_path}"
    
    def plot_learning_rate(self, save_path: Optional[str] = None) -> str:
        """绘制学习率曲线"""
        if not self.training_logs or "learning_rate" not in self.training_logs[0]:
            return "没有学习率信息，无法绘制学习率曲线"
        
        steps = [log["step"] for log in self.training_logs]
        learning_rates = [log["learning_rate"] for log in self.training_logs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, learning_rates)
        plt.title("Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            return f"学习率曲线已保存到: {save_path}"
        else:
            save_path = os.path.join(self.log_dir, f"lr_curve_{int(time.time())}.png")
            plt.savefig(save_path)
            return f"学习率曲线已保存到: {save_path}"
    
    def save_logs(self, file_name: Optional[str] = None) -> str:
        """保存训练日志到文件"""
        if not file_name:
            file_name = f"training_log_{int(time.time())}.json"
        
        log_path = os.path.join(self.log_dir, file_name)
        
        logs = {
            "training_logs": self.training_logs,
            "evaluation_logs": self.evaluation_logs,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.get_training_summary()
        }
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4)
            return f"日志已保存到: {log_path}"
        except Exception as e:
            return f"保存日志失败: {e}"
    
    def load_logs(self, file_path: str) -> bool:
        """从文件加载训练日志"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            self.training_logs = logs.get("training_logs", [])
            self.evaluation_logs = logs.get("evaluation_logs", [])
            self.start_time = logs.get("start_time")
            self.end_time = logs.get("end_time")
            
            return True
        except Exception as e:
            print(f"加载日志失败: {e}")
            return False
    
    def register_callback(self, callback_fn: Callable[[Dict], None]) -> None:
        """注册回调函数，用于实时更新UI"""
        self.callback_fn = callback_fn
    
    def update_ui(self, data: Dict) -> None:
        """更新UI，调用回调函数"""
        if hasattr(self, 'callback_fn') and self.callback_fn:
            self.callback_fn(data)


if __name__ == "__main__":
    # 测试训练监控功能
    monitor = TrainingMonitor()
    monitor.start_training()
    
    # 模拟训练过程
    for i in range(100):
        loss = 1.0 / (i + 1)
        lr = 0.001 * (1 - i/100)
        monitor.log_training_step(i, loss, lr)
        
        if i % 10 == 0:
            monitor.log_evaluation_step(i, {"accuracy": 0.5 + i/200, "perplexity": 10 - i/20})
    
    monitor.end_training()
    
    # 获取训练摘要
    summary = monitor.get_training_summary()
    print(f"训练摘要: {summary}")
    
    # 绘制损失曲线
    loss_curve_path = monitor.plot_training_loss()
    print(loss_curve_path)
    
    # 绘制学习率曲线
    lr_curve_path = monitor.plot_learning_rate()
    print(lr_curve_path)
    
    # 保存日志
    log_path = monitor.save_logs()
    print(log_path)