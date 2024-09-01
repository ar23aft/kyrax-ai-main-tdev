import cohere
import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Set up Cohere client
co = cohere.Client("YOUR_API_KEY_HERE")

def text_generation(user_message, temperature):
    prompt = f"""
Generate a response to the given message. Return the response.

## Examples
User Message: I'm joining a new startup called Co1t today. Could you help me write a short introduction message to my teammates?
Response To Query: Sure! Here's an introduction message you can say to your teammates: Hello team, it's a pleasure to meet you all. My name's [Your Name], and I look forward to collaborating with all of you, letting our ideas flow and tackling challenges. I'm excited to be part of the Co1t band! If you need any help, let me know and I'll try my best to assist you, anyway I can.

User Message: Make the introduction message sound more upbeat and conversational
Response To Query: Hey team, I'm stoked to be joining the Co1t crew today! Can't wait to dive in, get to know you all, and tackle the challenges and adventures that lie ahead together!

User Message: Give suggestions about the name of a Sports Application.
Response To Query: Certainly! Here are a few suggestions: MyFitnessPal, GymWingman, Muscle-ly, Cardiobro, ThePumper, GetMoving.

User Message: Give a startup idea for a Home Decor application.
Response To Query: Absolutely! An app that calculates the best position of your indoor plants for your apartment.
    
User Message: I have a startup idea about a hearing aid for the elderly that automatically adjusts its levels and with a battery lasting a whole week. Suggest a name for this idea
Response To Query: Hearspan

User Message: Give me a name of a startup idea regarding an online primary school that lets students mix and match their own curriculum based on their interests and goals
Response To Query: Prime Age

## Your Task
User Message: {user_message}
Response To Query:"""

    preamble ="""## Task & Context
    Generate concise responses, with a minimum of one sentence. 
    
    ## Style guide
    Be professional.
    """

    response = co.chat(
        query=prompt,
        model='command',
        temperature=temperature,
        preamble=preamble,
        max_tokens=150
    )

    return response.text.strip()

def convert_categorical_to_numerical(df):
    # Convert common categorical values
    df = df.applymap(lambda x: 1 if str(x).strip().lower() in ['yes', 'true', '1'] else 0 if str(x).strip().lower() in ['no', 'false', '0'] else x)
    
    # Convert remaining categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.Categorical(df[col]).codes
            
    return df

def decision_tree_classifier(df):
    df = df.fillna(0)  # Convert NaN or None to 0

    df = convert_categorical_to_numerical(df)

    features = [col for col in df.columns if col != 'Play']
    X = df[features]
    y = df['Play']

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    plt.figure(figsize=(15, 10))
    tree.plot_tree(dtree, feature_names=features, filled=True)
    plt.show()

def main():
    st.title("🚀 Kyrax A.I")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Choose a function", ["Chatbox", "Decision Tree Classification"])

    if option == "Chatbox":
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
                    "Creativity", value=0.5,
                    key="creativity_input",
                    min_value=0.1,
                    max_value=0.9,
                    help="Lower values generate more predictable output, higher values generate more creative output"
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

    elif option == "Decision Tree Classification":
        st.header("Decision Tree Classifier")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Original Data:")
            st.write(df)
            
            if st.checkbox("Convert non-numerical values to numerical"):
                st.write("Processed Data:")
                df_processed = df.copy()
                df_processed = df_processed.fillna(0)  # Handle NaN or None
                df_processed = convert_categorical_to_numerical(df_processed)
                st.write(df_processed)

                decision_tree_classifier(df_processed)

if __name__ == "__main__":
    main()
