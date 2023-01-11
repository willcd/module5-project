import pandas as pd
import re

# Prepping data for Question Quality model

from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text]

def clean_text(s):
    # remove code
    s = re.sub('<pre><code>.*?</code></pre>', '', s)
    # remove html tags
    s = re.compile(r'<.*?>').sub(r'', s)
    # remove links
    s = re.compile(r'https?://\S+|www\.\S+').sub(r'', s)
    # lowercase
    s = s.lower()
    s = utils.to_unicode(s)
    # implement gensim filters
    for f in filters:
        s = f(s)
    return s

def clean_text_60k(text):
    # remove code
    text = re.sub('<pre><code>.*?</code></pre>', '', text)
    # remove html tags
    text = re.compile(r'<.*?>').sub(r'', text)
    # remove links
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    # remove linefeeds, non-letter/numbers, and whitespace
    text = re.sub(r'\n',' ', text)
    text = re.sub(r'\W', " ", text)
    text = re.sub(" +", " ", text)
    return text

def create_and_clean_text_column(df, text_cleaner):
    df['text'] = df['Title'] + ' ' + df['Body']
    df['text'] = df['text'].apply(text_cleaner)
    return df

def quality_60k_full_clean():
    dirty_data_train = pd.read_csv("data/Quality_60k/train.csv")
    dirty_data_valid = pd.read_csv("data/Quality_60k/valid.csv")
    dirty_data = pd.concat([dirty_data_train, dirty_data_valid], ignore_index=True)
    dirty_data = create_and_clean_text_column(dirty_data, clean_text_60k)
    
    dirty_data['target'] = dirty_data['Y'].map(lambda x: 1 if x=='HQ' else 0)
    
    cleaned_data = dirty_data[['text','target']]
    
    cleaned_data.to_csv('data/Quality_60k/data_clean.csv')
    
    return cleaned_data

# Prepping Python Q&A data

def quality_python_full_clean():
    
    questions = pd.read_csv("data/QA_python/Questions.csv", encoding='latin1')
    answers = pd.read_csv("data/QA_python/Answers.csv", encoding='latin1')
    
    questions = questions.iloc[-100000:]
    answers = answers[answers['ParentId'].isin(questions['Id'])]
    
    questions = create_and_clean_text_column(questions, clean_text)
    
    questions['answer_count'] = [len(answers[answers['ParentId']==x]) for x in list(questions['Id'])]
    
    cleaned_data = questions[['text','answer_count', 'Score']]
    
    cleaned_data.to_csv('data/QA_python/data_clean.csv')
    
    return cleaned_data

# Prepping 10% of All Q&A data for tag prediction

def tags_full_clean():
    
    questions = pd.read_csv("data/QA_all/Questions.csv", encoding='latin1')
    tags = pd.read_csv("data/QA_all/Tags.csv", encoding='latin1')
    
    questions = questions.iloc[-200000:]
    tags = tags[tags['Id'].isin(questions['Id'])]
    
    questions = create_and_clean_text_column(questions, clean_text)
    
    questions['tags'] = questions['Id'].apply(lambda x: \
                                              ([str(y) for y in list(tags[tags['Id']==x]['Tag'])]))
    cleaned_data = questions[['Id', 'text', 'tags']]
        
    cleaned_data.to_csv('data/QA_all/data_clean.csv')
    
    return cleaned_data