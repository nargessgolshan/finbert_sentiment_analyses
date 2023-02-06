from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import torch

#name of the folder for your input and output

folder=r"J:\OneDrive - University of Kentucky\Github\Finbert"

#name of the input file
input_file=r"sentiment_analysis_example_input.csv"

#name of the column that you want classified

col="sentence"

#name of the output file

output_file="results.csv"


#no change to the code from here
df=pd.read_csv(folder+"\\"+input_file)


finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df=df[df[col].notnull()]
   
sentences = [str(x) for x in df[col]]


results = nlp(sentences)
print(results)  

results2 = pd.DataFrame.from_records(results)

final_results=pd.concat([df,results2],axis=1)

final_results.to_csv(folder+"\\"+output_file)
