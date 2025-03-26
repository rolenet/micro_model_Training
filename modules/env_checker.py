import os
import torch
import psutil
import platform
import socket
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Union

class EnvironmentChecker:
    """环境检测模块，用于验证当前机器是否满足微调要求"""
    
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.python_info = {}
        self.dependencies_info = {}
    
    def check_system(self) -> Dict:
        """检查系统信息"""
        self.system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
            "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
            "disk_total": round(psutil.disk_usage('/').total / (1024**3), 2),  # GB
            "disk_free": round(psutil.disk_usage('/').free / (1024**3), 2),  # GB
        }
        return self.system_info
    
    def check_gpu(self) -> Dict:
        """检查GPU信息"""
        self.gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if self.gpu_info["cuda_available"]:
            devices = []
            for i in range(self.gpu_info["gpu_count"]):
                device = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),  # GB
                    "memory_reserved": round(torch.cuda.memory_reserved(i) / (1024**3), 2),  # GB
                    "memory_allocated": round(torch.cuda.memory_allocated(i) / (1024**3), 2),  # GB
                }
                devices.append(device)
            self.gpu_info["devices"] = devices
        
        return self.gpu_info
    
    def check_python(self) -> Dict:
        """检查Python环境"""
        self.python_info = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch_version": torch.__version__,
        }
        return self.python_info
    
    def _check_package(self, package_name: str) -> Dict:
        """检查指定的包是否已安装
        
        Args:
            package_name: 包名称
            
        Returns:
            包含安装状态和版本信息的字典
        """
        try:
            # 尝试导入包
            module = __import__(package_name)
            # 获取版本信息
            version = getattr(module, '__version__', 'Unknown')
            return {
                "installed": True,
                "version": version
            }
        except ImportError:
            # 包未安装
            return {
                "installed": False,
                "version": None,
                "message": f"未安装{package_name}包"
            }
    
    def _check_package_with_details(self, package_name: str) -> Dict:
        """检查指定的包是否已安装，并提供更详细的错误信息
        
        Args:
            package_name: 包名称
            
        Returns:
            包含安装状态、版本信息和详细错误信息的字典
        """
        try:
            # 尝试导入包
            module = __import__(package_name)
            # 获取版本信息
            version = getattr(module, '__version__', 'Unknown')
            return {
                "installed": True,
                "version": version
            }
        except ImportError as e:
            # 包未安装
            error_msg = str(e)
            result = {
                "installed": False,
                "version": None,
                "message": f"未安装{package_name}包"
            }
            
            # 为特定包提供更详细的错误信息和解决建议
            if package_name == "unsloth" and "git" in error_msg.lower():
                result["error_details"] = "安装unsloth时Git克隆失败，可能是网络连接问题"
                result["solution"] = "请检查网络连接或代理设置，确保能够访问GitHub。也可以尝试使用镜像源安装或手动下载安装。"
            elif "No module named" in error_msg:
                result["solution"] = f"请使用pip安装: pip install {package_name}"
            
            return result
    
    def _check_soundfile_dependency(self) -> Dict:
        """检查soundfile库及其系统依赖是否正确安装
        
        Returns:
            包含安装状态、版本信息和详细错误信息的字典
        """
        result = {
            "installed": False,
            "version": None,
            "message": "未安装soundfile包"
        }
        
        try:
            import soundfile
            result["installed"] = True
            result["version"] = getattr(soundfile, "__version__", "Unknown")
            # 检查libsndfile是否可用
            try:
                result["libsndfile_version"] = soundfile.__libsndfile_version__
                # 尝试加载一个小音频文件以验证功能是否正常
                try:
                    # 仅尝试导入必要的函数，不实际读取文件
                    from soundfile import read as sf_read
                    result["functional"] = True
                except Exception as e:
                    result["functional"] = False
                    result["error_details"] = f"soundfile功能测试失败: {str(e)}"
                    result["solution"] = "虽然soundfile已安装，但功能测试失败，可能需要重新安装或检查系统依赖"
            except AttributeError:
                result["error_details"] = "soundfile已安装但无法获取libsndfile版本信息"
                result["solution"] = "可能需要重新安装soundfile或手动安装libsndfile库"
        except ImportError as e:
            error_msg = str(e)
            result["error_details"] = f"导入soundfile失败: {error_msg}"
            
            if "libsndfile" in error_msg.lower() or "dll" in error_msg.lower() or "cannot load library" in error_msg.lower():
                result["error_details"] = "缺少系统级依赖库libsndfile"
                if platform.system() == "Windows":
                    result["solution"] = "请下载并安装libsndfile: https://github.com/libsndfile/libsndfile/releases\n" \
                                        "1. 下载最新的Windows版本(例如libsndfile-1.2.0-win64.zip)\n" \
                                        "2. 解压后将libsndfile-1.dll重命名为libsndfile.dll\n" \
                                        "3. 将libsndfile.dll复制到Python安装目录或系统PATH目录\n" \
                                        "4. 重新启动应用程序\n" \
                                        "或者尝试: pip uninstall soundfile -y && pip install soundfile --upgrade"
                elif platform.system() == "Linux":
                    result["solution"] = "请使用系统包管理器安装libsndfile:\n" \
                                        "Ubuntu/Debian: sudo apt-get install libsndfile1\n" \
                                        "CentOS/RHEL: sudo yum install libsndfile"
                elif platform.system() == "Darwin":  # macOS
                    result["solution"] = "请使用Homebrew安装libsndfile: brew install libsndfile"
            elif "No module named" in error_msg:
                result["solution"] = "请使用pip安装: pip install soundfile>=0.12.1"
            else:
                result["solution"] = "请尝试重新安装soundfile: pip uninstall soundfile -y && pip install soundfile>=0.12.1"
        except Exception as e:
            result["error_details"] = f"检查soundfile时发生未知错误: {str(e)}"
            result["solution"] = "请尝试重新安装soundfile: pip uninstall soundfile -y && pip install soundfile>=0.12.1"
            
        return result
    
    def check_dependencies(self)-> Dict:
        """检查依赖包是否已安装"""
        dependencies = {
            "torch": self._check_package_with_details("torch"),
            "torchvision": self._check_package_with_details("torchvision"),
            "transformers": self._check_package_with_details("transformers"),
            "datasets": self._check_package_with_details("datasets"),
            "peft": self._check_package_with_details("peft"),
            "accelerate": self._check_package_with_details("accelerate"),
            "bitsandbytes": self._check_package_with_details("bitsandbytes"),
            "trl": self._check_package_with_details("trl"),
            "scipy": self._check_package_with_details("scipy"),
            "scikit-learn": self._check_package_with_details("scikit-learn"),
            "matplotlib": self._check_package_with_details("matplotlib"),
            "pandas": self._check_package_with_details("pandas"),
            "numpy": self._check_package_with_details("numpy"),
            "soundfile": self._check_soundfile_dependency(),
        }
        
        # 只有在有NVIDIA GPU的情况下才检查unsloth
        if self.gpu_info.get("cuda_available", False) and self.gpu_info.get("gpu_count", 0) > 0:
            dependencies["unsloth"] = self._check_package_with_details("unsloth")
            # 如果unsloth安装失败，标记为可选但提供详细错误信息
            if not dependencies["unsloth"].get("installed", False):
                dependencies["unsloth"]["optional"] = True  # 标记为可选依赖
        else:
            # 如果没有GPU，标记unsloth为不可用但不影响整体检查
            dependencies["unsloth"] = {
                "installed": False,
                "version": None,
                "optional": True,  # 标记为可选依赖
                "message": "Unsloth需要NVIDIA GPU才能运行"
            }
        
        return dependencies
    
    def check_all(self) -> Dict:
        """检查所有环境信息"""
        return {
            "system": self.check_system(),
            "gpu": self.check_gpu(),
            "python": self.check_python(),
            "dependencies": self.check_dependencies()
        }
    
    def is_suitable_for_training(self, min_memory_gb: float = 8.0, min_gpu_memory_gb: float = 8.0) -> Tuple[bool, List[str]]:
        """检查当前环境是否适合进行模型微调"""
        self.check_all()
        issues = []
        
        # 检查内存
        if self.system_info.get("memory_available", 0) < min_memory_gb:
            issues.append(f"可用内存不足: {self.system_info.get('memory_available')}GB < {min_memory_gb}GB")
        
        # 检查GPU
        if not self.gpu_info.get("cuda_available", False):
            issues.append("CUDA不可用，无法使用GPU进行训练")
        elif self.gpu_info.get("gpu_count", 0) == 0:
            issues.append("未检测到可用的GPU设备")
        else:
            # 检查GPU内存
            has_sufficient_gpu = False
            for device in self.gpu_info.get("devices", []):
                available_memory = device.get("memory_total", 0) - device.get("memory_allocated", 0)
                if available_memory >= min_gpu_memory_gb:
                    has_sufficient_gpu = True
                    break
            
            if not has_sufficient_gpu:
                issues.append(f"没有足够内存的GPU: 需要至少{min_gpu_memory_gb}GB可用GPU内存")
        
        # 检查依赖包
        missing_packages = []
        for package, info in self.dependencies_info.items():
            if not info.get("installed", False):
                missing_packages.append(package)
        
        if missing_packages:
            issues.append(f"缺少必要的依赖包: {', '.join(missing_packages)}")
        
        return len(issues) == 0, issues


if __name__ == "__main__":
    # 测试环境检测功能
    checker = EnvironmentChecker()
    env_info = checker.check_all()
    print("系统信息:", env_info["system"])
    print("GPU信息:", env_info["gpu"])
    print("Python信息:", env_info["python"])
    print("依赖包信息:", env_info["dependencies"])
    
    is_suitable, issues = checker.is_suitable_for_training()
    if is_suitable:
        print("当前环境适合进行模型微调")
    else:
        print("当前环境不适合进行模型微调，存在以下问题:")
        for issue in issues:
            print(f"- {issue}")