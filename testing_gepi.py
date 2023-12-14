import numpy as np
import torch


from modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    GPT2Config,
    GPT2Tokenizer
)

mode = "topic"
code_desired = "true"
code_undesired = "false"
model_type = 'gpt2'
gen_type = "gedi"
gen_model_name_or_path = "gpt2-xl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),}
config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, do_lower_case=False)
model = model_class.from_pretrained(gen_model_name_or_path, torch_dtype=torch.float16)
model = model.to(device)
model = model.float()

gedi_model_name_or_path = 'gedi_topic'
gedi_model = model_class.from_pretrained(gedi_model_name_or_path)
gedi_model.to(device)
#max generation length
gen_length = 50
#omega from paper, higher disc_weight means more aggressive topic steering
disc_weight = 30
#1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
filter_p = 0.8
#tau from paper, preserves tokens that are classified as correct topic
target_p = 0.8
#hyperparameter that determines class prior, set to uniform by default
class_bias = 0

if gen_length>1024:
  length = 1024
else:
  length = gen_length
secondary_code = 'climate'
bpe_tokens = tokenizer.encode(secondary_code)
if len(bpe_tokens) > 1:
  print("Warning! number of bpe tokens for " + secondary_code + " is greater than 1, model isn't trained for this, generation is less likely to match the topic")
prompt = "In a shocking finding"

start_len=0
text_ids = tokenizer.encode(prompt)
encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(device)

multi_code = tokenizer.encode(secondary_code)
attr_class = 1

generated_sequence = model.generate(input_ids=encoded_prompts,
                                         pad_lens=None,
                                          max_length= length,
                                          top_k=None,
                                          top_p=None,
                                          repetition_penalty= 1.2,
                                          rep_penalty_scale= 10,
                                          eos_token_ids = tokenizer.eos_token_id,
                                          pad_token_id = 0,
                                          do_sample= False,
                                          penalize_cond= True,
                                          gedi_model= gedi_model,
                                          tokenizer= tokenizer,
                                          disc_weight= disc_weight,
                                          filter_p = filter_p,
                                          target_p = target_p,
                                          class_bias = class_bias,
                                          attr_class = attr_class,
                                          code_0 = code_undesired,
                                          code_1 = code_desired,
                                          multi_code=multi_code
                                          )

text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True)
print('\n')
print(text)