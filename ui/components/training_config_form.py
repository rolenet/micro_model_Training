from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional

class TrainingConfigForm(FlaskForm):
    """微调配置表单"""
    # 基本训练参数
    batch_size = IntegerField('批次大小', validators=[DataRequired(), NumberRange(min=1)], default=4)
    learning_rate = FloatField('学习率', validators=[DataRequired(), NumberRange(min=0)], default=2e-4)
    num_epochs = IntegerField('训练轮数', validators=[DataRequired(), NumberRange(min=1)], default=3)
    max_steps = IntegerField('最大步数', validators=[Optional()], default=-1)
    gradient_accumulation_steps = IntegerField('梯度累积步数', validators=[DataRequired(), NumberRange(min=1)], default=1)
    warmup_ratio = FloatField('预热比例', validators=[DataRequired(), NumberRange(min=0, max=1)], default=0.03)
    weight_decay = FloatField('权重衰减', validators=[DataRequired(), NumberRange(min=0)], default=0.001)
    max_seq_length = IntegerField('最大序列长度', validators=[DataRequired(), NumberRange(min=1)], default=512)
    
    # LoRA参数
    lora_r = IntegerField('LoRA秩', validators=[DataRequired(), NumberRange(min=1)], default=8)
    lora_alpha = IntegerField('LoRA Alpha', validators=[DataRequired(), NumberRange(min=1)], default=16)
    lora_dropout = FloatField('LoRA Dropout', validators=[DataRequired(), NumberRange(min=0, max=1)], default=0.05)
    
    # 数据集配置
    text_column = StringField('文本列名', default="text")
    instruction_column = StringField('指令列名', default="instruction")
    response_column = StringField('回复列名', default="response")
    
    # 优化器设置
    optimizer = SelectField('优化器', choices=[
        ('adamw_hf', 'AdamW (Hugging Face)'),
        ('adamw_torch', 'AdamW (PyTorch)'),
        ('adafactor', 'Adafactor'),
        ('sgd', 'SGD'),
        ('adagrad', 'Adagrad')
    ], default='adamw_hf')
    
    scheduler = SelectField('学习率调度器', choices=[
        ('linear', 'Linear'),
        ('cosine', 'Cosine'),
        ('cosine_with_restarts', 'Cosine with Restarts'),
        ('polynomial', 'Polynomial'),
        ('constant', 'Constant'),
        ('constant_with_warmup', 'Constant with Warmup')
    ], default='cosine')
    
    # 系统参数
    save_steps = IntegerField('保存步数', validators=[DataRequired(), NumberRange(min=1)], default=500)
    eval_steps = IntegerField('评估步数', validators=[DataRequired(), NumberRange(min=1)], default=100)
    logging_steps = IntegerField('日志步数', validators=[DataRequired(), NumberRange(min=1)], default=10)
    output_dir = StringField('输出目录', default="./outputs")
    
    # 混合精度训练
    fp16 = BooleanField('FP16', default=True)
    bf16 = BooleanField('BF16', default=False)
    
    submit = SubmitField('保存配置')