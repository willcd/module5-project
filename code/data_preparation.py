import pandas as pd

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
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

def create_and_clean_text_column(df):
    df['text'] = df['Title'] + ' ' + df['Body']
    df['text'] = df['text'].apply(clean_text)
    return df

def quality_60k_full_clean():
    """
    This is the one function called that will run all the support functions.
    Assumption: 
        - Your data files will be saved in a data folder and named "dirty_data.csv"
        - OR you might read directly from a few urls
        - this code is guidance, not rules

    :return: cleaned dataset to be passed to hypothesis testing and visualization modules.
    """
    dirty_data = pd.read_csv("../data/Quality_60k/data.csv")
    
    dirty_data = create_and_clean_text_column(dirty_data)
    
    dirty_data['target'] = dirty_data['Y'].map(lambda x: 1 if x=='HQ' else 0)
    
    cleaned_data = dirty_data[['text','target']]
    
    cleaned_data.to_csv('../data/Quality_60k/data_clean.csv')
    
    return cleaned_data

# Prepping Python Q&A data

def quality_python_full_clean():
    
    questions = pd.read_csv("../data/QA_python/Questions.csv")
    answers = pd.read_csv("../data/QA_python/Answers.csv")
    
    questions = questions.iloc[-100000:]
    answers = answers[answers['ParentId'].isin(questions['Id'])]
    
    questions = create_and_clean_text_column(questions)
    
    questions['answer_count'] = [len(answers[answers['ParentId']==x]) for x in list(questions['Id'])]
    
    cleaned_data = questions[['text','answer_count', 'score']]
    
    cleaned_data.to_csv('../data/QA_python/data_clean.csv')
    
    return cleaned_data

# Prepping 10% of All Q&A data for tag prediction

def tags_full_clean():
    
    questions = pd.read_csv("../data/QA_all/Questions.csv")
    tags = pd.read_csv("../data/QA_all/Tags.csv")
    
    questions = questions.iloc[-200000:]
    tags = tags[tags['Id'].isin(questions['Id'])]
    
    questions = create_and_clean_text_column(questions)
    
    questions['tags'] = questions['Id'].apply(lambda x: \
                                              ([str(y) for y in list(tags[tags['Id']==x]['Tag'])]))
    cleaned_data = questions[['text', 'tags']]
        
    cleaned_data.to_csv('../data/QA_all/data_clean.csv')
    
    return cleaned_data