from datasets import load_from_disk
import pandas as pd
import time
from perspective import PerspectiveAPI
from tqdm import tqdm
import random
import argparse

API_KEY = 'AIzaSyDL1Edz1npW9iXNx1O8Pr0_Dm-lQESPk5g"'
Perspective = PerspectiveAPI(API_KEY)
def analyze_text(blocking_prompt):
    print('blocking_prompt',blocking_prompt)
    toxicity_score = None
    try:
        result = Perspective.score(blocking_prompt)
        toxicity_score = result.get("TOXICITY")
        print("Toxicity score: " + str(toxicity_score))

    except Exception as e:
        print(f"An error occurred: {e}")  # 打印错误消息

    finally:
        time.sleep(1)  # 确保在成功或失败的情况下都会暂停
        return toxicity_score  # 返回毒性评分，如果有错误或"TOXICITY"键不存在，则为None
def main(model_name, output_file, num_beams, threshold, max_tokens, method, prompt_eng, num_samples, collection_name,
         api, step_size, random_method,scaling_size,input_file):
    df= pd.read_csv(input_file)
    df['toxic_score'] = df['model_real_output'].apply(analyze_text)
    df.to_csv(input_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2-medium",
                        help="The name or path of the model")
    parser.add_argument("--output_file", type=str, required=False, help="The name of the output CSV file")
    parser.add_argument("--num_beams", type=int, default=20, help="The number of beams for beam search")
    parser.add_argument("--num_samples", type=int, default=20, help="The number of num_samples for sampling search")
    parser.add_argument("--threshold", type=float, default=0.3, help="The threshold")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max length")
    parser.add_argument("--method", type=str, default="random", help="method")
    parser.add_argument("--prompt_eng", type=bool, default=False, help="whether add hard prompt")
    parser.add_argument("--collection_name", type=str, default="toxic", help="data collection name")
    parser.add_argument("--API_KEY", type=str, default="AIzaSyDL1Edz1npW9iXNx1O8Pr0_Dm-lQESPk5g", help="API")
    parser.add_argument("--step_size", type=int, default=1, help="step size")
    parser.add_argument("--random_method", type=str, default=None, help="random method")
    parser.add_argument("--scaling_size", type=float, default=1.0, help="The scaling_size")
    parser.add_argument("--input_file",type=str, required=True, help="The name of the input CSV file")

    args = parser.parse_args()
    main(args.model_name, args.output_file, args.num_beams, args.threshold, args.max_tokens, args.method,
         args.prompt_eng, args.num_samples, args.collection_name, args.API_KEY, args.step_size, args.random_method,args.scaling_size,args.input_file)

