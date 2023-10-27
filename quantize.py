from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import pandas as pd
import argparse

def quantize(model_path,output_path, quant_config, data_path='./data/calibrate_data.csv',num_samples=1000):
    model = AutoAWQForCausalLM.from_pretrained(model_path,device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
   
    if 'csv' not in data_path:
        raise ValueError('Please provice a csv file as the calibrate data!  ')
    data = pd.read_csv(data_path)
    if 'text' not in data.columns:
        raise ValueError(''' The provided csv file must contain a 'text' column ''' )
    data = data.sample(frac=1.0)
    samples = data['text'].iloc[0:num_samples]
    model.quantize(tokenizer, quant_config=quant_config,calib_data=samples)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model file path")
    parser.add_argument("-o", "--output_path", help="output path")
    parser.add_argument("-d", "--data_path",default='./data/calibrate_data.csv', help="calibrate data file path")
    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    data_path = args.data_path
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }
    quantize(model_path,output_path, quant_config, data_path,num_samples=1000)
    
