import cohere
import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Set up Cohere client
co = cohere.Client("your_api_key_here")

def text_generation(user_message, temperature):
    """
    Generate a response based on the user's message and specified temperature.
    """
    prompt = f"""
Generate a response to the given message. Return the response.

## Examples
User Message: I'm joining a new startup called Co1t today. Could you help me write a short introduction message to my teammates?
Response To Query: Sure! Here's an introduction message you can say to your teammates: Hello team, it's a pleasure to meet you all. My name's [Your Name], and I look forward to collaborating with all of you, letting our ideas flow and tackling challenges. I'm excited to be part of the Co1t band! If you need any help, let me know and I'll try my best to assist you, anyway I can.

User Message: Make the introduction message sound more upbeat and conversational
Response To Query: Hey team, I'm stoked to be joining the Co1t crew today! Can't wait to dive in, get to know you all, and tackle the challenges and adventures that lie ahead together!

User Message: Give suggestions about the name of a Sports Application.
Response To Query: Certainly! Here are afew suggestions: MyFitnessPal, GymWingman, Muscle-ly, Cardiobro, ThePumper, GetMoving.

User Message: Give a startup idea for a Home Decor application.
Response To Query: Absolutely! An app that calculates the best position of your indoor plants for your apartment.

User Message: I have a startup idea about a hearing aid for the elderly that automatically adjusts its levels and with a battery lasting a whole week. Suggest a name for this idea
Response To Query: Hearspan

User Message: Give me a name of a startup idea regarding an online primary school that lets students mix and match their own curriculum based on their interests and goals
Response To Query: Prime Age

## Your Task
User Message: {user_message}
Response To Query:"""

    # Create a custom preamble:
    preamble = """## Task & Context
    Generate concise responses, with minimum one-sentence. 
    
    ## Style guide
    Be professional.
    """

    response = co.chat(
        query=prompt,
        model='command-xlarge-nightly',
        temperature=temperature,
        preamble=preamble
    )

    return response.text.strip()

# Streamlit app

def encode_data(df):
    # Replace NaN or None values with 0
    df.fillna(0, inplace=True)

    # Specific mapping for "yes"/"no" variants
    yes_no_map = {'yes': 1, 'YES': 1, 'Yes': 1, 'no': 0, 'NO': 0, 'No': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(yes_no_map).fillna(df[col])  # Apply specific mapping first

    # General encoding for other categorical columns
    df_encoded = df.copy()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            df_encoded[column] = df_encoded[column].astype('category').cat.codes

    return df_encoded

def decision_tree_classification(df):
    # Convert non-numeric columns
    df_encoded = encode_data(df)

    # Features and target variable
    features = df_encoded.columns[:-1]  # Assuming the last column is the target
    X = df_encoded[features]
    y = df_encoded[df_encoded.columns[-1]]

    # Decision Tree Classifier
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    # Plotting the tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(dtree, feature_names=features, filled=True)
    st.pyplot(plt)

# Streamlit UI

st.title("🚀 Kyrax A.I")

# Sidebar for feature selection
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Chatbox", "Decision Tree Classification"])

if app_mode == "Chatbox":
    st.header("Chatbox")

    form = st.form(key="user_settings")
    with form:
        st.write("Enter your query below [Example: Startup Idea, Introduction Message, Name Suggestions] ")
        query_input = st.text_input("User Message", key="message_input")

        col1, col2 = st.columns(2)
        with col1:
            num_input = st.slider(
                "Number of results", 
                value=3, 
                key="num_input", 
                min_value=1, 
                max_value=10,
                help="Choose to generate between 1 to 10 results"
            )
        with col2:
            creativity_input = st.slider(
                "Creativity", 
                value=0.5, 
                key="creativity_input", 
                min_value=0.1, 
                max_value=0.9,
                help="Lower values generate more 'predictable' output, higher values generate more 'creative' output"
            )

        generate_button = form.form_submit_button("Generate Result")

        if generate_button:
            if query_input == "":
                st.error("Input/Query field cannot be blank")
            else:
                my_bar = st.progress(0.05)
                st.subheader("Results:")

                for i in range(num_input):
                    st.markdown("""---""")
                    user_message = text_generation(query_input, creativity_input)
                    st.write(user_message)
                    my_bar.progress((i + 1) / num_input)

elif app_mode == "Decision Tree Classification":
    st.header("Decision Tree Classification")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data before Encoding:")
        st.write(df)

        decision_tree_classification(df)

        st.write("Data after Encoding:")
        st.write(encode_data(df))
