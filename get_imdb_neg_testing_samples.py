from datasets import load_dataset
dataset = load_dataset("imdb",split='test')
neg_reviews = dataset.filter(lambda x: x['label'] == 0)
neg_reviews = neg_reviews.select(range(2500))
neg_reviews.save_to_disk('imdb_neg_2500')
