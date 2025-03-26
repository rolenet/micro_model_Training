import os
import json
from typing import Dict, List, Optional, Union
from huggingface_hub import snapshot_download, list_datasets, login
from datasets import load_dataset, Dataset, DatasetDict

class DatasetManager:
    """数据集管理模块，用于下载和管理数据集"""
    
    def __init__(self, config_path: str = None):
        # 修改配置文件路径，使用项目根目录
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default_config.json")
        self.config = self._load_config(config_path)
        self.datasets_dir = os.path.abspath(self.config["dataset"]["local_datasets_dir"])
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.current_dataset = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "dataset": {
                    "default_dataset": "tatsu-lab/alpaca",
                    "local_datasets_dir": "./datasets"
                }
            }
    
    def list_local_datasets(self) -> List[str]:
        """列出本地已下载的数据集"""
        if not os.path.exists(self.datasets_dir):
            return []
        
        datasets = []
        for item in os.listdir(self.datasets_dir):
            item_path = os.path.join(self.datasets_dir, item)
            if os.path.isdir(item_path):
                datasets.append(item)
        
        return datasets
    
    def search_huggingface_datasets(self, query: str = "", task_type: str = None, limit: int = 20) -> List[Dict]:
        """搜索HuggingFace上的数据集"""
        try:
            datasets = list_datasets(filter=task_type, search=query, limit=limit)
            return [{
                "id": dataset.id,
                "downloads": dataset.downloads,
                "likes": dataset.likes,
                "tags": dataset.tags
            } for dataset in datasets]
        except Exception as e:
            print(f"搜索数据集失败: {e}")
            return []
    
    def download_dataset(self, dataset_id: str, use_auth_token: Optional[str] = None) -> str:
        """从HuggingFace下载数据集"""
        try:
            # 如果提供了token，先登录
            if use_auth_token:
                login(token=use_auth_token)
            
            # 下载数据集到本地
            dataset_dir = os.path.join(self.datasets_dir, dataset_id.split("/")[-1])
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                token=use_auth_token
            )
            
            return dataset_dir
        except Exception as e:
            print(f"下载数据集失败: {e}")
            return ""
    
    def load_dataset(self, dataset_path: str, split: str = None, subset: str = None) -> bool:
        """加载数据集"""
        try:
            # 检查路径是否为HuggingFace数据集ID
            if not os.path.exists(dataset_path) and "/" in dataset_path:
                print(f"数据集路径不存在，尝试从HuggingFace加载: {dataset_path}")
                try:
                    # 直接从HuggingFace加载
                    self.current_dataset = load_dataset(dataset_path, subset, split=split)
                    print(f"从HuggingFace加载数据集成功: {dataset_path}")
                    return True
                except Exception as e:
                    print(f"从HuggingFace加载数据集失败: {e}，尝试下载后加载")
                    dataset_path = self.download_dataset(dataset_path)
                    if not dataset_path:
                        return False
            
            # 从本地加载数据集
            if os.path.isdir(dataset_path):
                # 检查是否为HuggingFace格式的数据集目录
                if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
                    self.current_dataset = load_dataset(dataset_path, split=split)
                else:
                    # 尝试加载自定义格式的数据集
                    self.current_dataset = self._load_custom_dataset(dataset_path)
            elif os.path.isfile(dataset_path):
                # 根据文件扩展名加载数据集
                ext = os.path.splitext(dataset_path)[1].lower()
                if ext == ".json":
                    self.current_dataset = load_dataset("json", data_files=dataset_path, split=split)
                elif ext == ".csv":
                    self.current_dataset = load_dataset("csv", data_files=dataset_path, split=split)
                elif ext == ".txt":
                    self.current_dataset = load_dataset("text", data_files=dataset_path, split=split)
                elif ext == ".parquet":
                    self.current_dataset = load_dataset("parquet", data_files=dataset_path, split=split)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")
            else:
                raise ValueError(f"无效的数据集路径: {dataset_path}")
            
            print(f"加载数据集成功: {dataset_path}")
            return True
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return False
    
    def _load_custom_dataset(self, dataset_path: str) -> Union[Dataset, DatasetDict]:
        """加载自定义格式的数据集"""
        # 检查目录中的文件类型
        json_files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
        txt_files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
        parquet_files = [f for f in os.listdir(dataset_path) if f.endswith(".parquet")]
        
        # 优先选择一种文件类型
        if json_files:
            data_files = [os.path.join(dataset_path, f) for f in json_files]
            return load_dataset("json", data_files=data_files)
        elif csv_files:
            data_files = [os.path.join(dataset_path, f) for f in csv_files]
            return load_dataset("csv", data_files=data_files)
        elif parquet_files:
            data_files = [os.path.join(dataset_path, f) for f in parquet_files]
            return load_dataset("parquet", data_files=data_files)
        elif txt_files:
            data_files = [os.path.join(dataset_path, f) for f in txt_files]
            return load_dataset("text", data_files=data_files)
        else:
            raise ValueError(f"目录中没有找到支持的数据文件: {dataset_path}")
    
    def get_dataset_info(self) -> Dict:
        """获取当前加载的数据集信息"""
        if self.current_dataset is None:
            return {"error": "未加载数据集"}
        
        try:
            info = {}
            
            # 处理DatasetDict类型
            if isinstance(self.current_dataset, DatasetDict):
                info["type"] = "DatasetDict"
                info["splits"] = list(self.current_dataset.keys())
                info["num_splits"] = len(info["splits"])
                
                # 获取第一个split的信息作为示例
                first_split = info["splits"][0]
                dataset = self.current_dataset[first_split]
                info["first_split"] = {
                    "name": first_split,
                    "num_rows": len(dataset),
                    "features": list(dataset.features.keys()) if hasattr(dataset, "features") else [],
                    "sample": dataset[0] if len(dataset) > 0 else {}
                }
            
            # 处理Dataset类型
            elif isinstance(self.current_dataset, Dataset):
                info["type"] = "Dataset"
                info["num_rows"] = len(self.current_dataset)
                info["features"] = list(self.current_dataset.features.keys()) if hasattr(self.current_dataset, "features") else []
                info["sample"] = self.current_dataset[0] if len(self.current_dataset) > 0 else {}
            
            return info
        except Exception as e:
            return {"error": f"获取数据集信息失败: {e}"}
    
    def prepare_for_training(self, text_column: str = "text", instruction_column: str = None, 
                           response_column: str = None, split: str = "train") -> Dataset:
        """准备用于训练的数据集"""
        if self.current_dataset is None:
            raise ValueError("未加载数据集")
        
        # 获取正确的数据集对象
        dataset = self.current_dataset
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"数据集中不存在分割: {split}")
            dataset = dataset[split]
        
        # 检查列是否存在
        columns = dataset.column_names
        if text_column and text_column not in columns and instruction_column not in columns:
            raise ValueError(f"数据集中不存在列: {text_column} 或 {instruction_column}")
        
        return dataset


if __name__ == "__main__":
    # 测试数据集管理功能
    manager = DatasetManager()
    local_datasets = manager.list_local_datasets()
    print(f"本地数据集: {local_datasets}")
    
    # 搜索HuggingFace数据集
    hf_datasets = manager.search_huggingface_datasets(query="alpaca", limit=5)
    print(f"HuggingFace数据集搜索结果: {hf_datasets}")