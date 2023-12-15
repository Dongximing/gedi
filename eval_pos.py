# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(test_file, test_column):
    #pig4431/YELP_roBERTa_5E
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    sentence_list = pd.read_csv(test_file)
    model = model.to('cuda')

    predicted_classes =[]
    batch_size = 8
    cleaned_sentences = [s for s in sentence_list[test_column]]
    data_loader = DataLoader(cleaned_sentences, batch_size=batch_size)
    for batch in tqdm(data_loader):

        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_index = torch.argmax(probabilities, dim=-1)
        for index in predicted_class_index:
            predicted_class = 'positive' if index.item() == 1 else 'negative'
            predicted_classes.append(predicted_class)
    print(len(predicted_classes))
    #sentence_list['model_score'] = predicted_classes
    print("avg score", predicted_classes.count('positive') / len(predicted_classes))
    sentence_list.to_csv(test_file, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--test_file", type=str, required=True, help="file name")
    parser.add_argument("--test_column", type=str, default="completions", help="test column")
    args = parser.parse_args()
    main(args.test_file, args.test_column)
