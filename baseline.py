
import torch
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
# ds = load_dataset('imdb', split='test')
# ds = ds.select(range(2500))
ds = pd.read_csv('/mnt/hdd-data/shaowei/Ximing/gedi/sentiment_halfpos_halfneg_testing_gedi.csv')

ds = ds['prompt'].tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = model.to(device)
results = []
output_file = 'sentiment_halfpos_halfneg_gpt_med_testing_baseline.csv'
for i in tqdm(range(len(ds))):
    try:
        toxic_prompt = ds[i]
        text_ids = tokenizer.encode(toxic_prompt)
        encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(device)
        encoded_prompts = encoded_prompts[:,:8]
        input_size = len(encoded_prompts[0])
        print("length",input_size)
        start_time = time.time()
        gen_length=input_size+25
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                            max_length= gen_length,
                                            pad_token_id = tokenizer.eos_token_id
                                            )
        end_time = time.time()
        excution_time = end_time - start_time
        text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True)
        results.append(
            {'prompt': toxic_prompt, 'model_real_output': tokenizer.decode(generated_sequence.tolist()[0][input_size:],
                                                                           clean_up_tokenization_spaces=True),
             "completions": text, 'total_time': excution_time})
    except ValueError as e:
        print(e)
        continue



results_df = pd.DataFrame(results)
time= results_df['total_time'].mean()
print("time---------->",time)
results_df.to_csv(output_file, index=False)
print(f"All prompts have been processed and saved to {output_file}")
