from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import nltk.data
from nltk import tokenize
import string
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
import base64
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
import io 
import threading
matplotlib.use('agg')

app = Flask(__name__)
def count_sentences(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    return len(sentences)

def pos_filter(text):
    # Tokenize the input text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Perform POS tagging
    tagged_tokens = pos_tag(filtered_tokens)
    
    # Keep only nouns, verbs, adjectives, and adverbs
    pos_filtered_tokens = [word for word, pos in tagged_tokens if pos.startswith('N') or pos.startswith('V') or pos.startswith('J') or pos.startswith('R')]
    
    return ' '.join(pos_filtered_tokens)

def calculate_sentence_scores(sentences, freq_dist, stop_words):
    scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word not in stop_words:
                if sentence not in scores:
                    scores[sentence] = freq_dist[word]
                else:
                    scores[sentence] += freq_dist[word]
    return scores


def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    
    sentence_scores = calculate_sentence_scores(sentences, freq_dist, stop_words)
    
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_sentences = sorted_sentences[:num_sentences]
    
    summary = TreebankWordDetokenizer().detokenize(summary_sentences)
    return summary


def remove_plural(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    singular_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(singular_words)
def remove_duplicates(text):
    words = text.split()
    unique_words = []
    
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    
    return ' '.join(unique_words)

def extract_nouns_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    nouns = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    #print(nouns)
    return ' '.join(nouns)

def extract_verbs_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    verbs = [word for word, pos in tagged_tokens if pos in ['VB', 'VBD','VBG','VBN','VBP','VBZ']]
    #print (verbs)
    return ' '.join(verbs)

def extract_adjectives_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    adjectives = [word for word, pos in tagged_tokens if pos in ['JJ', 'JJR','JJS']]
    #print (adjectives)
    return ' '.join(adjectives)

def extract_adverbs_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    adverbs = [word for word, pos in tagged_tokens if pos in ['RB', 'RBR','RBS']]
    #print (adverbs)
    return ' '.join(adverbs)

def extract_nounsverbs_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    nounverbs = [word for word, pos in tagged_tokens if pos in ['VB', 'VBD','NN', 'NNS', 'NNP', 'NNPS']]
    #print(nounverbs)
    return ' '.join(nounverbs)
    
    #return nounverbs
    

# Store the plot in memory as a BytesIO object

 


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    year = int(request.form.get('year'))
    quarter = int(request.form.get('quarter'))

    df=pd.read_csv('transcripts_of_NTR.csv')
    df2=pd.read_csv('transcripts_list_of_NTRQuarterly.csv')
    df3=pd.merge(df, df2, on='id')
    df4=pd.DataFrame()
    df4=df3.copy()
    count=0
    for i in range(0,len(df4)):
        if (df4['year'][i]==year) and (df4['quarter'][i]==quarter) :
            print(f" {df4['year'][i]} {year}")
            count=count+1
    
    name="Data"+str(year)+"q"+str(quarter)+".csv"
    df4 = df4[df4['year'] == year]
    df4 = df4[df4['quarter'] == quarter]
    df4.to_csv(name)
    df=df4.copy()
    #capitalnamefiles is new
    
    text=""
    for i in range(0,len(df['speech'].index)):
        text=text+df['speech'][i]+"\n"
    nltk.download('punkt')  # Download the necessary data for tokenization

    nltk.download('punkt')
    text=text

    # Call the function to count sentences
    num_sentences = count_sentences(text)

    # Print the result
    num_summary_sentences = 5
    summary = summarize_text(text, num_sentences=num_summary_sentences)
    # Pre-process summary for generating word clouds
    output_text=remove_duplicates(remove_plural(extract_nounsverbs_nltk(summary)))
   
    return redirect(url_for('display', summary=summary,output_text=output_text))
    
    
@app.route('/display')
def display():
    summary = request.args.get('summary')
    output_text = request.args.get('output_text')
    print("phele", file=sys.stderr)
    print(output_text, file=sys.stderr)
    print("baadme", file=sys.stderr)

    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text)
    img_stream = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream, format="png")
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.read()).decode()
    return render_template('display.html', summary=summary,wordcloud_image=img_base64)
    

if __name__ == "__main__":
    app.run(debug=True)