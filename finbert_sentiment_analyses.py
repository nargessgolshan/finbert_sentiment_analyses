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
df.reset_index(drop=True,inplace=True)

for index,row in df.iterrows():
    text=str(row[col])
    sentiment=nlp(text)
    df.loc[index,'label']=sentiment[0]['label']
    df.loc[index,'score']=sentiment[0]['score']

df.to_csv(folder+"\\"+output_file,index=False)
