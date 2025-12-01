from pprint import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, device=-1)

text = "Thys is a pamphlet sentence."

result = nlp(text)  
pprint(result)
