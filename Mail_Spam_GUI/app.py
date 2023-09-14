import streamlit as st
from PIL import Image
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

img = Image.open('gmail-icon.png')
st.set_page_config(page_title='Spam Detection', page_icon=img)
hide_menu_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Detection")
st.write("\n")
st.write("\n")
print("\n")
st.subheader("Enter the msg:")
input_sms = st.text_area("")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.sidebar.image("gmail-icon.png", width=100, use_column_width='always')
st.sidebar.header("Options")
button11 = st.sidebar.button("Examples of Spam/Ham Mails")
if button11:
    st.write("\n\n\n")
    st.write("\n")
    st.write("\n")
    st.write("Some examples of Spam and Ham mails:")
    give_examples = """
        <table border="5" width="100%" border-color="black">
        <tr>
        <th>Sr.No.</th>
        <th>Text</th>
        <th>Result</th> 
        </tr>
        <tr>
        <td>1</td>
        <td>WINNER!! As a valued network customer you have been selected to received 9000 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.</td>
        <td bgcolor="#C13A50">Spam</td>
        </tr>
        <tr>
        <td>2</td>
        <td>I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? </td>
        <td bgcolor="green">Ham</td>
        </tr>
        <tr>
        <td>3</td>
        <td>Congratulations! You have won a lottery!</td>
        <td bgcolor="#C13A50">Spam</td>
        </tr>
        <tr>
        <td>4</td>
        <td>Hello, saw your presentation today, would like to discuss few things with you</td>
        <td bgcolor="green">Ham</td>
        </tr>
        <tr>
        <td>5</td>
        <td>Thanks for your subscription to Ringtone UK your mobile will be charged 5/month Please confirm by replying YES or NO. If you reply NO you will not be charged</td>
        <td bgcolor="#C13A50">Spam</td>
        </tr>
        <tr>
        <td>6</td>
        <td>Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo. Not that i'm trying to invite myself or anything!</td>
        <td bgcolor="green">Ham</td>
        </tr>  
        </table>
        """
    st.markdown(give_examples, unsafe_allow_html=True)

button12 = st.sidebar.button("Comparision of Algorithms")
if button12:
    st.write("\n")
    st.write("Comparision of various Algorithms:")
    st.image("download.png", width=700)

button13 = st.sidebar.button("Ham Words")
if button13:
    st.write("\n")
    st.write("Ham Words are:")
    st.image("hamwords(1).png", width=500)

button14 = st.sidebar.button("Spam Words")
if button14:
    st.write("\n")
    st.write("Spam Words are:")
    st.image("spamwords(1).png", width=500)
