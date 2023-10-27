### Before run the code, please revise the 'model_type' field in the 'config.json' file into 'llama' 
### 运行前需要把模型文件中的'config.json'配置文件中的'model_type' 改成llama
### Or you can just use the config.json file in this repo to replace the one in the model file folder 
### 或者你也可以用本项目中的config.json文件来覆盖模型文件中的config.json


### Quantization 

-  In the project folder
-  try: python quantize.py -m my/model/file/path -o my/output/file/path

### Inference

```
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized('/home/ouwei/baai/baai-awq/',trust_remote_code=True,fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained('/home/ouwei/baai/baai-awq/',trust_remote_code=True)
prompt ='''### Human : 写一个杭州旅游攻略 \n ### Assistant:'''
input = tokenizer(prompt,return_tensor='pt')
print(model.generate(**input))
```