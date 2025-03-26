from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Optional

class DatasetSelectorForm(FlaskForm):
    """数据集选择表单"""
    dataset_source = SelectField('数据集来源', choices=[
        ('local', '本地数据集'),
        ('huggingface', 'HuggingFace数据集'),
        ('custom', '自定义数据集')
    ], default='local')

    local_path = StringField('本地数据集路径', validators=[Optional()])
    dataset_id = StringField('HuggingFace数据集ID', validators=[DataRequired()])
    custom_path = StringField('自定义数据集路径', validators=[Optional()])
    # 添加缺少的config_name字段
    config_name = StringField('配置名称', validators=[Optional()])
    local_path = StringField('本地数据集路径')
    subset = StringField('数据集子集 (可选)')
    split = StringField('数据集分割 (可选，默认为train)')
    use_auth_token = StringField('HuggingFace认证Token（私有数据集需要）', validators=[Optional()])
    submit = SubmitField('加载数据集')