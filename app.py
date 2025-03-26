import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from modules.env_checker import EnvironmentChecker
from modules.model_manager import ModelManager
from modules.dataset_manager import DatasetManager
from modules.training_config import TrainingConfig
from modules.training_monitor import TrainingMonitor
from ui.components.model_selector import ModelSelectorForm
from ui.components.dataset_selector import DatasetSelectorForm
from ui.components.training_config_form import TrainingConfigForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化各个模块
env_checker = EnvironmentChecker()
model_manager = ModelManager()
dataset_manager = DatasetManager()
training_config = TrainingConfig()
training_monitor = TrainingMonitor()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/env_check')
def env_check():
    """环境检测页面"""
    try:
        env_checker = EnvironmentChecker()
        env_info = env_checker.check_all()
        
        # 确定环境是否适合微调
        is_suitable = True
        issues = []
        
        # 检查系统内存
        if env_info["system"]["memory_total"] < 8:
            is_suitable = False
            issues.append("系统内存不足，建议至少8GB内存")
        
        # 检查磁盘空间
        if env_info["system"]["disk_free"] < 10:
            is_suitable = False
            issues.append("可用磁盘空间不足，建议至少10GB可用空间")
        
        # 检查GPU
        if not env_info["gpu"]["cuda_available"]:
            # 不将GPU缺失作为致命问题，但添加警告
            issues.append("未检测到NVIDIA GPU，某些功能可能受限或性能较低")
        
        # 检查必要依赖
        for package, info in env_info["dependencies"].items():
            if not info["installed"] and not info.get("optional", False):
                is_suitable = False
                issues.append(f"缺少必要依赖: {package}")
        
        return render_template('env_check.html', 
                              env_info=env_info, 
                              is_suitable=is_suitable, 
                              issues=issues)
    except Exception as e:
        # 捕获所有异常，确保页面能够正常显示
        return render_template('env_check.html', 
                              env_info={"system": {}, "gpu": {}, "python": {}, "dependencies": {}}, 
                              is_suitable=False, 
                              issues=[f"环境检测出错: {str(e)}"])

@app.route('/model_management', methods=['GET', 'POST'])
def model_management():
    """模型管理页面"""
    form = ModelSelectorForm()
    local_models = model_manager.list_local_models()
    model_loaded = False
    model_info = None
    
    if form.validate_on_submit():
        if form.model_source.data == 'huggingface':
            model_path = form.model_id.data
        else:
            model_path = form.local_path.data
        
        # 加载模型
        if form.use_unsloth.data:
            model_loaded = model_manager.load_model_with_unsloth(
                model_path, 
                load_in_8bit=form.load_in_8bit.data,
                load_in_4bit=form.load_in_4bit.data
            )
        else:
            model_loaded = model_manager.load_model(
                model_path, 
                load_in_8bit=form.load_in_8bit.data,
                load_in_4bit=form.load_in_4bit.data
            )
        
        if model_loaded:
            flash(f'模型 {model_path} 加载成功！', 'success')
            model_info = model_manager.get_model_info(model_path)
        else:
            flash(f'模型 {model_path} 加载失败！', 'error')
    
    # 搜索HuggingFace模型
    hf_models = []
    if request.args.get('search'):
        query = request.args.get('search')
        hf_models = model_manager.search_huggingface_models(query=query, limit=10)
    
    return render_template('model_management.html', form=form, local_models=local_models, 
                           hf_models=hf_models, model_loaded=model_loaded, model_info=model_info)

@app.route('/dataset_management', methods=['GET', 'POST'])
def dataset_management():
    """数据集管理页面"""
    form = DatasetSelectorForm()
    local_datasets = dataset_manager.list_local_datasets()
    dataset_loaded = False
    dataset_info = None
    
    if form.validate_on_submit():
        if form.dataset_source.data == 'huggingface':
            dataset_path = form.dataset_id.data
        else:
            dataset_path = form.local_path.data
        
        # 加载数据集
        dataset_loaded = dataset_manager.load_dataset(
            dataset_path, 
            split=form.split.data, 
            subset=form.subset.data
        )
        
        if dataset_loaded:
            flash(f'数据集 {dataset_path} 加载成功！', 'success')
            dataset_info = dataset_manager.get_dataset_info()
        else:
            flash(f'数据集 {dataset_path} 加载失败！', 'error')
    
    # 搜索HuggingFace数据集
    hf_datasets = []
    if request.args.get('search'):
        query = request.args.get('search')
        hf_datasets = dataset_manager.search_huggingface_datasets(query=query, limit=10)
    
    return render_template('dataset_management.html', form=form, local_datasets=local_datasets, 
                           hf_datasets=hf_datasets, dataset_loaded=dataset_loaded, dataset_info=dataset_info)

@app.route('/training_config', methods=['GET', 'POST'])
def training_config():
    """训练配置页面"""
    from ui.components.training_config_form import TrainingConfigForm
    
    # 创建配置管理器实例
    config_manager = TrainingConfig()
    
    form = TrainingConfigForm()
    config_saved = False
    
    # 如果是POST请求并且表单验证通过
    if form.validate_on_submit():
        # 保存配置
        training_config = {
            "batch_size": form.batch_size.data,
            "learning_rate": form.learning_rate.data,
            "num_epochs": form.num_epochs.data,
            "max_steps": form.max_steps.data,
            "gradient_accumulation_steps": form.gradient_accumulation_steps.data,
            "warmup_ratio": form.warmup_ratio.data,
            "weight_decay": form.weight_decay.data,
            "lora_r": form.lora_r.data,
            "lora_alpha": form.lora_alpha.data,
            "lora_dropout": form.lora_dropout.data,
            "optimizer": form.optimizer.data,
            "scheduler": form.scheduler.data,
            "save_steps": form.save_steps.data,
            "eval_steps": form.eval_steps.data,
            "logging_steps": form.logging_steps.data,
            "output_dir": form.output_dir.data,
            "fp16": form.fp16.data,
            "bf16": form.bf16.data,
            "max_seq_length": form.max_seq_length.data,
            "text_column": form.text_column.data,
            "instruction_column": form.instruction_column.data,
            "response_column": form.response_column.data
        }
        
        # 更新配置
        config_manager.update_training_params(training_config)
        config_manager.save_config()
        
        config_saved = True
    
    # 如果是GET请求，从配置文件加载当前配置
    else:
        # 获取当前训练参数
        current_config = config_manager.training_params
        
        # 填充表单
        if current_config:
            form.batch_size.data = current_config.get("batch_size", 4)
            form.learning_rate.data = current_config.get("learning_rate", 2e-4)
            form.num_epochs.data = current_config.get("num_epochs", 3)
            form.max_steps.data = current_config.get("max_steps", -1)
            form.gradient_accumulation_steps.data = current_config.get("gradient_accumulation_steps", 1)
            form.warmup_ratio.data = current_config.get("warmup_ratio", 0.03)
            form.weight_decay.data = current_config.get("weight_decay", 0.001)
            form.lora_r.data = current_config.get("lora_r", 8)
            form.lora_alpha.data = current_config.get("lora_alpha", 16)
            form.lora_dropout.data = current_config.get("lora_dropout", 0.05)
            form.optimizer.data = current_config.get("optimizer", "adamw_hf")
            form.scheduler.data = current_config.get("scheduler", "cosine")
            form.save_steps.data = current_config.get("save_steps", 500)
            form.eval_steps.data = current_config.get("eval_steps", 100)
            form.logging_steps.data = current_config.get("logging_steps", 10)
            form.output_dir.data = current_config.get("output_dir", "./outputs")
            form.fp16.data = current_config.get("fp16", True)
            form.bf16.data = current_config.get("bf16", False)
            form.max_seq_length.data = current_config.get("max_seq_length", 512)
            form.text_column.data = current_config.get("text_column", "text")
            form.instruction_column.data = current_config.get("instruction_column", "instruction")
            form.response_column.data = current_config.get("response_column", "response")
    
    return render_template('training_config.html', form=form, config_saved=config_saved)

@app.route('/start_training')
def start_training():
    """开始训练页面"""
    # 检查模型和数据集是否已加载
    if model_manager.current_model is None:
        flash('请先加载模型！', 'error')
        return redirect(url_for('model_management'))
    
    if dataset_manager.current_dataset is None:
        flash('请先加载数据集！', 'error')
        return redirect(url_for('dataset_management'))
    
    # 获取训练参数
    training_args = training_config.get_training_arguments()
    lora_config = training_config.get_lora_config()
    
    # 这里应该启动训练进程，但为了简化，我们只返回训练页面
    # 实际应用中，应该使用后台任务或WebSocket实时更新训练状态
    return render_template('training.html', training_args=training_args, lora_config=lora_config)

@app.route('/training_status')
def training_status():
    """训练状态API"""
    # 获取训练摘要
    summary = training_monitor.get_training_summary()
    return jsonify(summary)

@app.route('/training_logs')
def training_logs():
    """训练日志页面"""
    return render_template('training_logs.html')

@app.route('/api/training_logs')
def api_training_logs():
    """训练日志API"""
    logs = {
        'training_logs': training_monitor.training_logs,
        'evaluation_logs': training_monitor.evaluation_logs
    }
    return jsonify(logs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)