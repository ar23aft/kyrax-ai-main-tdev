import pandas as pd
import cohere
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# Set up Cohere client
co = cohere.Client("YOUR_API_KEY_HERE")  # Replace with your Cohere API key

def text_generation(user_message, temperature):
    """
    Generate a response using the Cohere API.

    Arguments:
        user_message (str): The user query or request.
        temperature (float): The creativity level for the model.

    Returns:
        str: The generated response.
    """
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

    preamble = """## Task & Context
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


# Function to automatically convert non-numerical values to numerical
def auto_convert(df):
    df = df.applymap(lambda x: 1 if str(x).strip().lower() in ['yes', 'true', '1'] else 0 if str(x).strip().lower() in ['no', 'false', '0'] else x)
    
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            df[column] = df[column].map(mapping)
    
    df.fillna(0, inplace=True)  # Convert NaN or None to 0
    return df

def decision_tree_classifier(df):
    """
    Create and display a Decision Tree classifier from the given DataFrame with customized colors.

    Arguments:
        df (pd.DataFrame): The DataFrame to process.
    """
    df = auto_convert(df)
    features = df.columns[:-1].tolist()
    X = df[features]
    y = df[df.columns[-1]]

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    fig, ax = plt.subplots(figsize=(15, 10))
    tree.plot_tree(dtree, feature_names=features, filled=False, ax=ax)

    # Customize colors
    for i, (node, decision) in enumerate(zip(dtree.tree_.feature, dtree.tree_.threshold)):
        color = 'lightblue' if decision < 0 else 'palevioletred' if dtree.tree_.value[i, 0, 0] > 0 else 'brown'
        ax.patches[i].set_facecolor(color)

    # Custom color for the topmost node (root node)
    root_patch = patches.FancyBboxPatch(
        (0.5, 0.5),  # Example coordinates; adjust as needed
        width=1, height=1,
        boxstyle="round,pad=0.1",
        edgecolor="none",
        facecolor="purple"
    )
    ax.add_patch(root_patch)

    st.pyplot(fig)

def main():
    st.title("🚀 Kyrax A.I")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Choose a function", ["Chatbox", "Decision Tree Classification"])

    if option == "Chatbox":
        form = st.form(key="user_settings")
        with form:
            st.write("Enter your query below [Example: Startup Idea, Introduction Message, Name Suggestions]")
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
            
            st.write("Converted Data:")
            df_processed = auto_convert(df)
            st.write(df_processed)

            decision_tree_classifier(df_processed)

if __name__ == "__main__":
    main()
