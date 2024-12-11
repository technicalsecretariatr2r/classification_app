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


st.set_page_config(
    page_title="RtR Categorization APP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_glossary():
    file_path_glossary = "glossary.csv"
    df_glossary = pd.read_csv(file_path_glossary, sep=';') 
    return df_glossary

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
df_glossary = load_glossary()



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



def display_app_introduction():
    st.title("🌍 Welcome to the Solution Mapping Tool")
    st.caption("Experimental Version")
    
    st.subheader("Challenge")
    st.markdown("""
    **Context**: Bringing together existing adaptation and resilience solutions into the RPI Solution Hub requires defining a conceptual framework that establishes meaningful connections. 
    Originally, a taxonomy of solutions was intended to serve this purpose. However, it was recognized that a single taxonomy cannot accommodate all the classification systems already 
    in use. As a result, the challenge remains open.
    """)
    
    st.subheader("Data and Approach")
    st.markdown("""
    RPI has chosen a minimalist approach, where the primary focus is on the description of solutions and their associated keywords. Based solely on this information, the goal is to identify:
    - The type of solution.
    - The types of actions involved.
    - The systems where the solution applies (e.g., food and agriculture, infrastructure, oceans and nature, finance, etc.).
    - Other potential classifications that connect solutions from the local to the global scale.

    For example, solutions applied locally can be mapped to their corresponding Sharm El Sheikh Adaptation Agenda Impact Systems or other classification frameworks. 
    This app serves as an **experimental tool** to achieve these connections, relying on the descriptions of the solutions.
    """)

display_app_introduction()

st.expander("About This Application", expanded=True).markdown("""
### What is This?
This application is a **Natural Language Processing (NLP) tool** designed to classify and analyze text related to adaptation and resilience solutions. It categorizes input text into predefined systems like SAA Priority Groups, Resilience Attributes, and more, while also providing insights and glossaries for further exploration.

---

### How is it Done?
- **Machine Learning:** The app uses **Naive Bayes classifiers** from TextBlob, trained on curated datasets for specific taxonomy systems.
- **Streamlit Framework:** The web interface is powered by Streamlit, enabling an interactive and user-friendly experience.
- **NLP Libraries:** Libraries like **TextBlob** and **NLTK** handle text processing, classification, sentiment analysis, and spell checking.
- **Data Handling:** Training and test data are loaded efficiently using **Pandas** and cached to optimize performance.

---

### What Can It Be Used For?
- **Classification:** Categorize adaptation and resilience solution descriptions into:
  - SAA Priority Groups
  - Resilience Attributes
  - Resilience Sub-Categories
  - Action Types and Clusters
  - Permaculture Flower Domains
- **Exploration:** Understand and explore key concepts using the glossary feature.
- **Analysis:** Perform text analysis tasks like sentiment detection, text statistics, and spell checking.
- **Learning Tool:** Demonstrates NLP techniques applied to sustainability and resilience contexts.

---

### Key Features
1. **Multi-Category Classification:**
   - Input a text description to classify it into relevant categories with associated probabilities.
   - View detailed classification results across various systems in a tabbed interface.

2. **Glossary Search:**
   - Explore terms, definitions, and categories related to the classifications.
   - Access external resources for deeper understanding.

3. **Custom Interface:**
   - Designed for a seamless user experience, with hidden Streamlit branding and optimized layouts.
   - Tabbed structure for clear navigation across classification systems.

---

This tool is experimental and tailored for researchers, analysts, and anyone working with sustainability and resilience solutions.
""")



st.subheader("Categorize Adaptation and Resilience Solutions")

st.info("""
**Instructions:**

Copy and paste the description of an adaptation and resilience solution, then press `Ctrl+Enter` to obtain the categories with the highest probabilities based on a trained dataset designed specifically for each system of categories (taxonomies). 

After obtaining the results, you can explore the meaning of any concept in the results using the glossary options below.
""")




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



@st.fragment
def glossary():
    st.subheader("Glossary Search")

    # Multiselection for filtering by categories
    selected_categories = st.multiselect(
        "Filter by categories:",
        options=df_glossary["Category"].unique(),
        # default=df_glossary["Category"].unique()  # Default selects all categories
    )

    # Filter the DataFrame based on selected categories
    filtered_df = df_glossary[df_glossary["Category"].isin(selected_categories)]

    # Display results
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            st.markdown(f"**Source:** {row['Source']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"**Definition:** {row['Definition']}")
            if row['Link']:
                st.markdown(f"[Learn more]({row['Link']})")
            st.markdown("---")
    else:
        st.write("No results found. Try adjusting your search or filters.")


glossary()
