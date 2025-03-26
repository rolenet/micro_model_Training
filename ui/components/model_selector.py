from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class ModelSelectorForm(FlaskForm):
    """模型选择表单"""
    model_source = SelectField('模型来源', choices=[
        ('huggingface', 'HuggingFace模型'),
        ('local', '本地模型')
    ])
    model_id = StringField('HuggingFace模型ID', validators=[DataRequired()])
    local_path = StringField('本地模型路径')
    use_auth_token = StringField('HuggingFace Token (如需访问私有模型)')
    load_in_8bit = BooleanField('使用8-bit量化')
    load_in_4bit = BooleanField('使用4-bit量化')
    use_unsloth = BooleanField('使用Unsloth加速')
    submit = SubmitField('加载模型')