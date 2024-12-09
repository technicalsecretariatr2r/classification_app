import streamlit as st
import nltk
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

nltk_data_dir = "./nltk_data/"
nltk.data.path.append(nltk_data_dir)


# nltk_data_dir = "./nltk_data/"
# nltk.data.path.clear()
# nltk.data.path.append(nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('wordnet', download_dir=nltk_data_dir)



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



st.set_page_config(
    page_title="RtR Categorization APP",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Filter RA data
df_ra_training = df_ra_training[~df_ra_training['level'].isin(["1", "2", "3"])]

# Initialize classifiers
@st.cache_data
def initialize_classifiers(df_saa_training, df_ra_training, df_actions_training, df_permacultura_training):
    # Create classifiers for each dataset
    cl_saa = NaiveBayesClassifier(list(df_saa_training[['text', 'priority_systems']].itertuples(index=False, name=None)))
    cl_ra = NaiveBayesClassifier(list(df_ra_training[['text', 'ra']].itertuples(index=False, name=None)))
    cl_sub_ra = NaiveBayesClassifier(list(df_ra_training[['text', 'sub_ra']].itertuples(index=False, name=None)))
    cl_actions = NaiveBayesClassifier(list(df_actions_training[['text', 'action']].itertuples(index=False, name=None)))
    cl_action_clusters = NaiveBayesClassifier(list(df_actions_training[['text', 'action cluster']].itertuples(index=False, name=None)))
    cl_action_types = NaiveBayesClassifier(list(df_actions_training[['text', 'action_type']].itertuples(index=False, name=None)))
    cl_permacultura = NaiveBayesClassifier(list(df_permacultura_training[['text', 'domain']].itertuples(index=False, name=None)))
    
    return {
        "cl_saa": cl_saa,
        "cl_ra": cl_ra,
        "cl_sub_ra": cl_sub_ra,
        "cl_actions": cl_actions,
        "cl_action_clusters": cl_action_clusters,
        "cl_action_types": cl_action_types,
        "cl_permacultura": cl_permacultura,
    }

classifiers = initialize_classifiers(df_saa_training, df_ra_training, df_actions_training, df_permacultura_training)

# User Input
st.title("RtR Categorization APP")
input_text = st.text_area(
    "Enter text to classify:",
    placeholder="Type something here...",
    height=150
)

if not input_text.strip():
    st.warning("Please enter text for classification!")

# Tabbed Classifier Interface
tabs = st.tabs([
    "SAA Priority Groups", 
    "Resilience Attributes", 
    "Resilience Sub-Categories", 
    "Actions", 
    "Action Clusters", 
    "Action Types", 
    "Permaculture Flower"
])

# Function to Display Classification Results
def result_classify(cl, input_text, classifier_name, tab):
    with tab:
        if input_text.strip():
            st.subheader(f"Classification Results for {classifier_name}")
            probabilities = cl.prob_classify(input_text)

            # Display top categories sorted by probability
            relevant_categories = sorted(
                [
                    {"Class": label, "Probability": probabilities.prob(label)}
                    for label in probabilities.samples() if probabilities.prob(label) > 0.01
                ],
                key=lambda x: x["Probability"],
                reverse=True
            )
            if relevant_categories:
                st.write("**Top Categories (Sorted by Probability):**")
                for category in relevant_categories:
                    st.write(f"- **{category['Class']}** (Probability: {category['Probability']:.3f})")
            else:
                st.info("No categories found with probability > 0.01.")

            # Expand to view all probabilities
            with st.expander(f"Full Probabilities for {classifier_name}"):
                prob_df = pd.DataFrame([
                    {"Class": label, "Probability": probabilities.prob(label)}
                    for label in probabilities.samples()
                ]).sort_values(by="Probability", ascending=False)
                st.table(prob_df)

# Classification Results in Tabs
result_classify(classifiers["cl_saa"], input_text, "SAA Priority Groups", tabs[0])
result_classify(classifiers["cl_ra"], input_text, "Resilience Attributes", tabs[1])
result_classify(classifiers["cl_sub_ra"], input_text, "Resilience Sub-Categories", tabs[2])
result_classify(classifiers["cl_actions"], input_text, "Actions", tabs[3])
result_classify(classifiers["cl_action_clusters"], input_text, "Action Clusters", tabs[4])
result_classify(classifiers["cl_action_types"], input_text, "Action Types", tabs[5])
result_classify(classifiers["cl_permacultura"], input_text, "Permaculture Flower Domain", tabs[6])

# Footer
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .footer {
        visibility: visible;
        text-align: center;
        color: #888;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Made with ❤️ using Streamlit | © 2024
    </div>
    """,
    unsafe_allow_html=True
)
