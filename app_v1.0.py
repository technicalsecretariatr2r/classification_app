import streamlit as st
import nltk
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

def example_nltk():


    # Training Data: 5 Categories
    train_data = [
        ("Hospitals are crucial for improving health outcomes.", "Health"),
        ("Vaccinations reduce the spread of diseases.", "Health"),
        ("A balanced diet includes fruits, vegetables, and proteins.", "Food"),
        ("Restaurants must ensure food safety standards.", "Food"),
        ("Roads and bridges are essential for transportation.", "Infrastructure"),
        ("Better public infrastructure improves urban living.", "Infrastructure"),
        ("Marine biodiversity is declining due to pollution.", "Oceans"),
        ("Ocean currents regulate the global climate.", "Oceans"),
        ("Affordable housing is necessary for growing populations.", "Human Settlement"),
        ("Urbanization impacts natural habitats.", "Human Settlement")
    ]

    # Testing Data
    test_data = [
        ("Healthcare facilities are vital for community well-being.", "Health"),
        ("Eating healthy food prevents chronic illnesses.", "Food"),
        ("Good infrastructure supports economic development.", "Infrastructure"),
        ("The oceans absorb large amounts of carbon dioxide.", "Oceans"),
        ("Housing policies must address overcrowding in cities.", "Human Settlement")
    ]

    # Train the Naive Bayes Classifier
    classifier = NaiveBayesClassifier(train_data)

    # Streamlit App
    st.title("Multi-Category Text Classifier")
    st.write("Classify text into one of five categories: Health, Food, Infrastructure, Oceans, or Human Settlement.")

    # User Input
    input_text = st.text_area("Enter text to classify:", placeholder="Type something here...")

    # Classify Button
    if st.button("Classify"):
        if input_text.strip():
            # Classify the input text
            blob = TextBlob(input_text, classifier=classifier)
            classification = blob.classify()

            # Display the classification
            st.subheader("Result")
            st.write(f"**Classified as:** {classification}")
        else:
            st.warning("Please enter some text to classify.")

    # Show Classifier Accuracy
    if st.checkbox("Show Classifier Accuracy on Test Data"):
        accuracy = classifier.accuracy(test_data)
        st.write(f"Classifier Accuracy: {accuracy:.2f}")

    # Display Training Data
    if st.checkbox("Show Training Data"):
        st.subheader("Training Data")
        st.write(train_data)

    # Display Test Data
    if st.checkbox("Show Test Data"):
        st.subheader("Test Data")
        st.write(test_data)
# example_nltk()

def example_use_textblob():
    # Download NLTK resources (ensure this is done before running the app)
    nltk.download('punkt')
    nltk.download('wordnet')

    # App Title
    st.title("NLP Application with TextBlob")
    st.write("A simple Natural Language Processing app for text analysis.")

    # Sidebar
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose an option:", ["Home", "Sentiment Analysis", "Text Statistics", "Spell Checker"])

    # Home Page
    if options == "Home":
        st.subheader("Welcome to the NLP App")
        st.write("""
        Use the navigation menu to:
        - Analyze sentiment
        - Get text statistics
        - Check and correct spelling
        """)

    # Sentiment Analysis
    elif options == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        input_text = st.text_area("Enter text to analyze sentiment:")
        
        if st.button("Analyze Sentiment"):
            if input_text.strip():
                blob = TextBlob(input_text)
                sentiment = blob.sentiment
                st.write("**Polarity:**", sentiment.polarity)
                st.write("**Subjectivity:**", sentiment.subjectivity)
            else:
                st.warning("Please enter some text!")

    # Text Statistics
    elif options == "Text Statistics":
        st.subheader("Text Statistics")
        input_text = st.text_area("Enter text to analyze statistics:")
        
        if st.button("Get Statistics"):
            if input_text.strip():
                blob = TextBlob(input_text)
                st.write("**Number of Words:**", len(blob.words))
                st.write("**Number of Sentences:**", len(blob.sentences))
                st.write("**Noun Phrases:**", list(blob.noun_phrases))
            else:
                st.warning("Please enter some text!")

    # Spell Checker
    elif options == "Spell Checker":
        st.subheader("Spell Checker")
        input_text = st.text_area("Enter text to check and correct spelling:")
        
        if st.button("Check Spelling"):
            if input_text.strip():
                blob = TextBlob(input_text)
                corrected_text = blob.correct()
                st.write("**Original Text:**", input_text)
                st.write("**Corrected Text:**", corrected_text)
            else:
                st.warning("Please enter some text!")

    # Footer
    st.sidebar.markdown("### About")
    st.sidebar.write("This NLP app is powered by [TextBlob](https://textblob.readthedocs.io/) and built with [Streamlit](https://streamlit.io/).")
# example_use_textblob()

def structure_and_format():
    im = Image.open("images/header.png")
    # st.set_page_config(page_title="SAA Priority Systems Classification APP", layout="wide", initial_sidebar_state="expanded")
    st.set_page_config(page_title="SAA Priority Systems Classification APP", initial_sidebar_state="expanded")
    st.logo(im, size="large")
    css_path = "style.css"

    with open(css_path) as css:
        st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
    ##
    ##Hide footer streamlit
    ##
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    ## Hide index when showing a table. CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    ##
    ## Inject CSS with Markdown
    ##
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
# structure_and_format()

def render_main_page(): 
    logo_path = "images/header.png"  

    col1, col2, col3 = st.columns([2,0.5,0.5])
    with col1: 
        st.image(logo_path) 
# render_main_page()




# Load data with caching
@st.cache_data
def load_solution_stories():
    file_path_solution_stories = "solution_stories_rtr_241105.csv"
    df_solution_stories = pd.read_csv(file_path_solution_stories, sep=';') 
    return df_solution_stories

@st.cache_data
def load_training_saa_priority_systems():
    file_path_df_saa_training = "saa_app_training.csv"
    df_saa_training = pd.read_csv(file_path_df_saa_training, sep=';') 
    return df_saa_training

@st.cache_data
def load_training_resilience_attributes():
    file_path_df_ra_training = "ra_app_training.csv"
    df_ra_training = pd.read_csv(file_path_df_ra_training, sep=';') 
    return df_ra_training

@st.cache_data
def load_training_actions():
    file_path_df_actions_training = "actions_app_training.csv"
    df_actions_training = pd.read_csv(file_path_df_actions_training, sep=';') 
    return df_actions_training

@st.cache_data
def load_training_permacultura():
    file_path_df_permacultura = "permaculture_app_training.csv"
    df_permacultura_training = pd.read_csv(file_path_df_permacultura, sep=';') 
    return df_permacultura_training


# Load datasets
df_solution_stories = load_solution_stories()
df_saa_training = load_training_saa_priority_systems()
df_ra_training = load_training_resilience_attributes()
df_actions_training = load_training_actions()
df_permacultura_training = load_training_permacultura()

df_ra_training = df_ra_training[df_ra_training['level'] != "1"]
df_ra_training = df_ra_training[df_ra_training['level'] != "2"]
df_ra_training = df_ra_training[df_ra_training['level'] != "3"]


@st.cache_data
def initialize_classifiers(
    df_saa_training, 
    df_ra_training, 
    df_actions_training, 
    df_permacultura_training
):
    """
    Initializes NaiveBayes classifiers for various datasets.
    
    Args:
        df_saa_training: DataFrame containing SAA training data.
        df_ra_training: DataFrame containing RA and SUBRA training data.
        df_actions_training: DataFrame containing ACTIONS, ACTION CLUSTERS, and ACTION TYPES training data.
        df_permacultura_training: DataFrame containing PERMACULTURA training data.
    
    Returns:
        A dictionary with initialized classifiers and their corresponding training data.
    """
    # SAA
    train_saa = list(df_saa_training[['text', 'priority_systems']].itertuples(index=False, name=None))
    cl_saa = NaiveBayesClassifier(train_saa)

    # RA
    train_ra = list(df_ra_training[['text', 'ra']].itertuples(index=False, name=None))
    cl_ra = NaiveBayesClassifier(train_ra)

    # SUBRA
    train_sub_ra = list(df_ra_training[['text', 'sub_ra']].itertuples(index=False, name=None))
    cl_sub_ra = NaiveBayesClassifier(train_sub_ra)

    # ACTIONS
    train_actions = list(df_actions_training[['text', 'action']].itertuples(index=False, name=None))
    cl_actions = NaiveBayesClassifier(train_actions)

    # ACTION CLUSTERS
    train_action_clusters = list(df_actions_training[['text', 'action cluster']].itertuples(index=False, name=None))
    cl_action_clusters = NaiveBayesClassifier(train_action_clusters)

    # ACTION TYPES
    train_action_types = list(df_actions_training[['text', 'action_type']].itertuples(index=False, name=None))
    cl_action_types = NaiveBayesClassifier(train_action_types)

    # PERMACULTURA
    train_permacultura = list(df_permacultura_training[['text', 'domain']].itertuples(index=False, name=None))
    cl_permacultura = NaiveBayesClassifier(train_permacultura)

    # Return all classifiers and their training data in a dictionary
    return {
        "train_saa": train_saa,
        "cl_saa": cl_saa,
        "train_ra": train_ra,
        "cl_ra": cl_ra,
        "train_sub_ra": train_sub_ra,
        "cl_sub_ra": cl_sub_ra,
        "train_actions": train_actions,
        "cl_actions": cl_actions,
        "train_action_clusters": train_action_clusters,
        "cl_action_clusters": cl_action_clusters,
        "train_action_types": train_action_types,
        "cl_action_types": cl_action_types,
        "train_permacultura": train_permacultura,
        "cl_permacultura": cl_permacultura,
    }

classifiers = initialize_classifiers(
    df_saa_training, 
    df_ra_training, 
    df_actions_training, 
    df_permacultura_training
)

train_saa = classifiers["train_saa"]
cl_saa = classifiers["cl_saa"]

train_ra = classifiers["train_ra"]
cl_ra = classifiers["cl_ra"]

train_sub_ra = classifiers["train_sub_ra"]
cl_sub_ra = classifiers["cl_sub_ra"]

train_actions = classifiers["train_actions"]
cl_actions = classifiers["cl_actions"]

train_action_clusters = classifiers["train_action_clusters"]
cl_action_clusters = classifiers["cl_action_clusters"]

train_action_types = classifiers["train_action_types"]
cl_action_types = classifiers["cl_action_types"]

train_permacultura = classifiers["train_permacultura"]
cl_permacultura = classifiers["cl_permacultura"]


st.title("RtR Categorization APP")

# User Input
input_text = st.text_area(
    "Enter text to classify:",
    placeholder="Type something here...",
    height=200  
)


def result_classify(cl, train, input_text, classifier_name):
    key = f"classify_{classifier_name}"  # Create a unique and informative key
    if st.button(f"Classify {classifier_name}", key=key):
        if input_text.strip():
            # Classify the input text
            probabilities = cl.prob_classify(input_text)
            
            # Extract and sort relevant categories (probabilities > 0.01)
            relevant_categories = sorted(
                [
                    {"Class": label, "Probability": probabilities.prob(label)}
                    for label in probabilities.samples() if probabilities.prob(label) > 0.01
                ],
                key=lambda x: x["Probability"],
                reverse=True
            )
            
            # Display the classification
            st.subheader(f"Classification Result for {classifier_name}")
            if relevant_categories:
                formatted_categories = [
                    f"{item['Class']} (Probability: {item['Probability']:.3f})"
                    for item in relevant_categories
                ]
                st.write(f"**Relevant Categories:** {', '.join(formatted_categories)}")
            else:
                st.write("No categories have a probability above the threshold of 0.01.")
            
            # Display full class probability table
            with st.expander(f"Read Full Class Probabilities for {classifier_name}"):
                st.subheader(f"Class Probabilities for {classifier_name}")
                prob_data = [
                    {"Class": label, "Probability": probabilities.prob(label)}
                    for label in probabilities.samples()
                ]
                prob_df = pd.DataFrame(prob_data).sort_values(by="Probability", ascending=False)
                st.table(prob_df)
            
            # Display classifier accuracy
            accuracy = cl.accuracy(train)
            st.subheader(f"Classifier Accuracy for {classifier_name}")
            st.write(f"Accuracy on training data: **{accuracy:.2f}**")
        else:
            st.warning("Please enter some text to classify.")

result_classify(cl_saa, train_saa, input_text, classifier_name="SAA Priority Groups")
result_classify(cl_ra, train_ra, input_text, classifier_name="Resilience Attributes")
result_classify(cl_sub_ra, train_sub_ra, input_text, classifier_name="Resilience Attributes Sub-Categories")
result_classify(cl_actions, train_actions, input_text, classifier_name="Actions")
result_classify(cl_action_clusters, train_action_clusters, input_text, classifier_name="Action Clusters")
result_classify(cl_action_types, train_action_types, input_text, classifier_name="Action Types")
result_classify(cl_permacultura, train_permacultura, input_text, classifier_name="Permacultura Flower Domain")