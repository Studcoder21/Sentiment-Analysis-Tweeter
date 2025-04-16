import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from wordcloud import WordCloud
from PIL import Image
import gdown
# Page Configuration
st.set_page_config(
    page_title="BERT Tweet Analysis Dashboard",
    page_icon="📊",
)

# ---- HEADERS ----

# Title
st.title("🔍 BERT-Based Tweet Sentiment Analysis Dashboard")

# Introduction Section
st.header("📌 What Is This Tool Used For?")
st.markdown("""
This application performs **sentiment analysis** on tweet data uploaded by the user. 
It uses **BERT (Bidirectional Encoder Representations from Transformers)** as the model.
After categorization, tweets are classified into positive, negative and neutral sentiment categories.
""")

# Model Description
st.header("🧠 Model Used: google-bert/bert-base-cased")
st.markdown("""
This project uses the `bert-base-cased` model trained by **Google**.  
The case-sensitive version performs better in areas like proper noun recognition and emphasis detection.

**Why `bert-base-cased`?**
- **Analyzes sentence context bidirectionally**
- Case sensitivity allows better detection of **irony, emphasis and special word usage**
- Effective even on small datasets

Model source: [google-bert/bert-base-cased](https://huggingface.co/google/bert-base-cased)
""")

# Model Training Results
st.header("📊 Model Training Results")

accuracy = 0.8656
precision = 0.8682
recall = 0.8656
f1_score = 0.8651

col1, col2 = st.columns(2)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy:.4f}")
    st.metric("🔍 Precision", f"{precision:.4f}")

with col2:
    st.metric("📢 Recall", f"{recall:.4f}")
    st.metric("✅ F1 Score", f"{f1_score:.4f}")

st.markdown("""
These metrics indicate the model's performance on real-world data:
- **Accuracy**: Proportion of all correct predictions
- **Precision**: How many positive predictions were correct
- **Recall**: How many true positives were identified
- **F1 Score**: Balanced average of precision and recall
""")

# Confusion Matrix (replace with your actual image path)
st.subheader("📉 Confusion Matrix")
st.image("cm.png", caption="Model Confusion Matrix", use_container_width=True)


# Challenges Section
st.header("⚠️ Challenges in Sentiment Analysis")
st.markdown("""
- **Irony and Sarcasm**: Indirect meanings are common in tweets.
- **Short and context-free sentences**: The 512 token limit can limit context.
- **Tweets with multiple emotions**: Mixed emotions are difficult to separate.
- **Emojis, hashtags and abbreviations**: Although BERT recognizes these structures, additional preprocessing may be required for interpretation.

Therefore, the `bert-base-cased` model, which can analyze the context very well, is preferred.
""")

# --- TEXT PROCESSING ---
# Abbreviation dictionary
abbreviations = {
    "can't":"can not",
    "don't":"do not",
    "doesn't":"does not",
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}



# Sentiment Mapping
sentiment_labels = {
    0: "Negative",
    1: "Neutral", 
    2: "Positive"
}

# Initialize counters
counters = {
    'hashtags': 0,
    'mentions': 0,
    'urls': 0,
    'emojis': 0,
    'abbreviations': 0
}

def reset_counters():
    for key in counters:
        counters[key] = 0

def clean_tweet(tweet):
    if isinstance(tweet, float):  # Handle NaN
        return ""
    
    global counters
    # Count and remove hashtags
    hashtags = re.findall(r"#\w+", tweet)
    counters['hashtags'] += len(hashtags)
    temp = re.sub(r"#\w+", "", tweet)
    
    # Count and remove mentions
    mentions = re.findall(r"@\w+", temp)
    counters['mentions'] += len(mentions)
    temp = re.sub(r"@\w+", "", temp)
    
    # Count and remove URLs
    urls = re.findall(r"http\S+|www\S+|https\S+", temp)
    counters['urls'] += len(urls)
    temp = re.sub(r"http\S+|www\S+|https\S+", "", temp, flags=re.MULTILINE)
    
    # Count and replace abbreviations
    words = temp.split()
    abbrev_count = sum(1 for word in words if word.lower() in abbreviations)
    counters['abbreviations'] += abbrev_count
    temp = ' '.join([abbreviations.get(word.lower(), word) for word in words])
    
    return " ".join(temp.split())  # Remove extra spaces

# Function to convert emojis to text
def convert_emojis_to_text(text):
    if pd.isna(text):  # Handle NaN values
        return text
    if not isinstance(text, str):  # Ensure it's a string
        return str(text)
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    counters['emojis'] += len(emoji_list)
    return emoji.demojize(text, language="en")

# ---- MODEL ----
#@st.cache_resource
###def load_model():
#    model_name = "final_model"
#    model = AutoModelForSequenceClassification.from_pretrained(model_name)
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    return model, tokenizer###
@st.cache_resource
def load_model():
    # Baixar modelo do Google Drive
    model_url = "https://drive.google.com/uc?id=1BUh-SLS8_8H5V4ID4sm6T0kY4h-KNaH8"
    gdown.download(model_url, model_zip, quiet=False)
    
    
    # Carregar modelo
    model = AutoModelForSequenceClassification.from_pretrained("final_model")
    tokenizer = AutoTokenizer.from_pretrained("final_model")
    return model, tokenizer
model, tokenizer = load_model()

def predict_sentiment(text):
    text = clean_tweet(text)
    text = convert_emojis_to_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy().flatten()
    sentiment = np.argmax(probs)
    return sentiment_labels[sentiment], probs

# ---- STREAMLIT APP ----
analysis_type = st.radio("Select analysis type:", ["Single Text Analysis", "File Upload"])

if analysis_type == "Single Text Analysis":
    user_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        if user_input.strip():
            reset_counters()
            
            st.subheader("Original Text")
            st.write(user_input)
            
            cleaned_text = clean_tweet(user_input)
            processed_text = convert_emojis_to_text(cleaned_text)
            
            st.subheader("Cleaned Text")
            st.write(processed_text)
            
            st.subheader("Cleaning Statistics")
            stats_df = pd.DataFrame.from_dict(counters, orient='index', columns=['Count'])
            st.table(stats_df)
            
            sentiment, probs = predict_sentiment(user_input)
            
            st.subheader("Analysis Results")
            st.write(f"**Sentiment:** {sentiment}")
            
            prob_df = pd.DataFrame({
                'Sentiment': list(sentiment_labels.values()),
                'Probability': probs
            })
            st.bar_chart(prob_df.set_index('Sentiment'))
            
        else:
            st.warning("Please enter some text")

elif analysis_type == "File Upload":
    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("**Uploaded Data Preview:**", df.head())

        columns = st.multiselect("Select column(s) to analyze:", df.columns)

        if st.button("Analyze"):
            if columns:
                sentiment_results = {}
                cleaning_stats = defaultdict(int)

                for col in columns:
                    reset_counters()
                    
                    st.write(f"📌 **Analyzing Column:** {col}")
                    
                    df[f"{col}_processed"] = df[col].dropna().astype(str).apply(clean_tweet)
                    df[f"{col}_processed"] = df[f"{col}_processed"].apply(convert_emojis_to_text)
                    
                    cleaning_stats[col] = counters.copy()
                    
                    st.subheader("Text Cleaning Sample")
                    st.write(df[[col, f"{col}_processed"]].head())
                    
                    st.subheader("Cleaning Statistics")
                    col_stats = pd.DataFrame.from_dict(counters, orient='index', columns=['Count'])
                    st.table(col_stats)
                    
                    sentiments = df[f"{col}_processed"].apply(lambda text: predict_sentiment(text)[0])
                    sentiment_counts = sentiments.value_counts()
                    sentiment_results[col] = sentiment_counts

                    # Visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[0], palette="coolwarm")
                    axes[0].set_title(f"Sentiment Distribution - {col}")
                    axes[1].pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", 
                               colors=sns.color_palette("coolwarm", len(sentiment_counts)))
                    axes[1].set_title(f"Sentiment Proportions - {col}")
                    st.pyplot(fig)

                    # Word Cloud
                    st.subheader("🔤 Word Cloud")
                    all_text = " ".join(df[f"{col}_processed"].dropna().tolist())
                    if all_text.strip():
                        wordcloud = WordCloud(width=800, height=400, background_color='white',
                                            colormap='coolwarm').generate(all_text)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(plt)
                    else:
                        st.info("No valid text for word cloud")

                # Overall stats
            

            else:
                st.warning("Please select at least one column!")
