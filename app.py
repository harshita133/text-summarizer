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
app.secret_key = 'abcd'
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


 
cache = {}

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('upload.html')

temp={}
@app.route('/upload', methods=['POST'])
def upload():
    year = int(request.form.get('year'))
    quarter = int(request.form.get('quarter'))
    company=(request.form.get('company'))
    commodity=(request.form.get('Commodity'))
    df=pd.read_csv('transcripts_of_MOS.csv')
    df2=pd.read_csv('transcripts_list_of_MOS.csv')
    df5=pd.read_csv('transcripts_list_of_NTRQuarterly.csv')
    df6=pd.read_csv('transcripts_of_NTR.csv')
    df3=pd.merge(df, df2, on='id')
    df7=pd.merge(df5, df6, on='id')
    df4=pd.DataFrame()
    if company=="MOS":
     df4=df3.copy()
    else:
     df4=df7.copy()
    
    count=0
    for i in range(0,len(df4)):
        if (df4['year'][i]==year) and (df4['quarter'][i]==quarter) :
            count=count+1
    
    name=company+"Data"+str(year)+"q"+str(quarter)+".csv"
    df4 = df4[df4['year'] == year]
    df4 = df4[df4['quarter'] == quarter]
    df4 = df4.assign(row_number=range(len(df4)))
    df4.set_index("row_number", inplace=True)
    df4.to_csv(name)
    df=df4.copy()
    #capitalnamefiles is new
    
    text=""
    for i in range(0,len(df['speech'].index)):
        text=text+df['speech'][i]+"\n"
    nltk.download('punkt')  # Download the necessary data for tokenization

    nltk.download('punkt')
    text=text
    temp['text']=text
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
    original_text = temp['text']
    
    output_text_adjectives=remove_duplicates(remove_plural(extract_adjectives_nltk(summary)))
    output_text_adverbs=remove_duplicates(remove_plural(extract_adverbs_nltk(summary)))
    output_text_nouns=remove_duplicates(remove_plural(extract_nouns_nltk(summary)))
    output_text_nounsverbs=remove_duplicates(remove_plural(extract_nounsverbs_nltk(summary)))
    output_text_verbs=remove_duplicates(remove_plural(extract_verbs_nltk(summary)))
    
    print("phele", file=sys.stderr)
    
    print(temp['text'], file=sys.stderr)
    print("baadme", file=sys.stderr)
    
    
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text)
    img_stream = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream, format="png")
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.read()).decode()
    
    wordcloud_verbs = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text_verbs)
    img_stream_verbs = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_verbs, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream_verbs, format="png")
    img_stream_verbs.seek(0)
    img_base64_verbs = base64.b64encode(img_stream_verbs.read()).decode()
    
    wordcloud_nounsverbs = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text_nounsverbs)
    img_stream_nounsverbs = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_nounsverbs, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream_nounsverbs, format="png")
    img_stream_nounsverbs.seek(0)
    img_base64_nounsverbs = base64.b64encode(img_stream_nounsverbs.read()).decode()
    
    wordcloud_nouns = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text_nouns)
    img_stream_nouns = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_nouns, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream_nouns, format="png")
    img_stream_nouns.seek(0)
    img_base64_nouns = base64.b64encode(img_stream_nouns.read()).decode()
    
    wordcloud_adverbs = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text_adverbs)
    img_stream_adverbs = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_adverbs, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream_adverbs, format="png")
    img_stream_adverbs.seek(0)
    img_base64_adverbs = base64.b64encode(img_stream_adverbs.read()).decode()
    
    wordcloud_adjectives = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(output_text_adjectives)
    img_stream_adjectives = io.BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_adjectives, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_stream_adjectives, format="png")
    img_stream_adjectives.seek(0)
    img_base64_adjectives = base64.b64encode(img_stream_adjectives.read()).decode()
    

    #original text is the main trasnrcipt data , serached_text are the output sentences after searching
    return render_template('display.html', original_text=original_text,summary=summary,searched_text="NA",wordcloud_image=img_base64,wordcloud_image_adjectives=img_base64_adjectives, wordcloud_image_adverbs=img_base64_adverbs , wordcloud_image_verbs=img_base64_verbs, wordcloud_image_nouns=img_base64_nouns,wordcloud_image_nounsverbs=img_base64_nounsverbs)


import re

def find_sentences_with_word(paragraph, word):
    # Split the paragraph into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)

    # Initialize a list to store sentences containing the word
    sentences_with_word = []

    # Iterate through each sentence
    for sentence in sentences:
        # Check if the word is in the sentence (case insensitive)
        if re.search(rf'\b{re.escape(word)}\b', sentence, re.IGNORECASE):
            sentences_with_word.append(sentence)
    
    return sentences_with_word

def have_same_starting_words(sentence1, sentence2):
    words1 = sentence1.split()[:5]
    words2 = sentence2.split()[:5]

    return words1 == words2

def remove_sentences_with_similar_starts(paragraph):
    sentences = paragraph.split(". ")
    sentences_to_remove = set()

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if have_same_starting_words(sentences[i], sentences[j]):
                sentences_to_remove.add(j)

    sentences_to_keep = [sentences[i] for i in range(len(sentences)) if i not in sentences_to_remove]
    cleaned_paragraph = ". ".join(sentences_to_keep)

    return cleaned_paragraph

@app.route('/search', methods=['POST'])
def search():
    word_to_find = request.form.get('word')
    wordcloud_image = request.form.get('wordcloud_image')
    wordcloud_image_adjectives = request.form.get('wordcloud_image_adjectives')
    wordcloud_image_adverbs = request.form.get('wordcloud_image_adverbs')
    wordcloud_image_verbs = request.form.get('wordcloud_image_verbs')
    wordcloud_image_nouns = request.form.get('wordcloud_image_nouns')
    wordcloud_image_nounsverbs = request.form.get('wordcloud_image_nounsverbs')
    summary = request.form.get('summary')
    original_text = request.form.get('original_text')

    sentences_containing_word = find_sentences_with_word(original_text, word_to_find)
    texttry = ' '.join(sentences_containing_word)
    cleaned_paragraph = remove_sentences_with_similar_starts(texttry)
    sentences_containing_word = sent_tokenize(cleaned_paragraph)
    print(f" {sentences_containing_word}")
    return render_template('display.html', searched_text=sentences_containing_word,original_text=original_text,summary=summary,wordcloud_image=wordcloud_image,wordcloud_image_adjectives=wordcloud_image_adjectives, wordcloud_image_adverbs=wordcloud_image_adverbs , wordcloud_image_verbs=wordcloud_image_verbs, wordcloud_image_nouns=wordcloud_image_nouns,wordcloud_image_nounsverbs=wordcloud_image_nounsverbs)


if __name__ == "__main__":
    app.run(debug=True)