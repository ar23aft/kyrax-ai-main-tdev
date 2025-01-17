import cohere
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import streamlit as st

# Set up Cohere client
co = cohere.Client("wRb0iELnAbGjPMMA0fxkMk3YSdU1MApV5SlG5Z4q")  # Replace with your Cohere API key

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
        message=prompt,
        model='command',
        temperature=temperature,
        preamble=preamble,
        max_tokens=150
    )

    return response.text.strip()

def auto_convert(df):
    """
    Automatically convert non-numerical values to numerical.

    Arguments:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame with numerical values.
    """
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
    Create and display a Decision Tree classifier from the given DataFrame.

    Arguments:
        df (pd.DataFrame): The DataFrame to process.
    """
    df = auto_convert(df)

    features = df.columns[:-1].tolist()
    X = df[features]
    y = df[df.columns[-1]]

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    plt.figure(figsize=(15, 10))
    tree.plot_tree(dtree, feature_names=features, filled=True)
    st.pyplot(plt)

def main():
    st.title("🚀 Kyrax A.I 🤖")
    st.subheader("A Prototype Assistant of Streamline Simplicity")
    st.write("###### Though still in early stages with limited features, it's committed to providing the best assistance possible!")
    # st.write("###### Given its few features, it is in the early stages of development but will try its best to help you out!")
    # With its limited features, it's still in early development, but it's dedicated to assisting you to the best of its ability!

    # Sidebar for navigation
    st.sidebar.title("Navigation Panel")
    st.sidebar.write("How may Kyrax help you?")
    option = st.sidebar.selectbox("Choose a function", ["Chatbox", "Decision Tree Classification", "Classifiers for Prediction"])
    st.sidebar.write("###### More features coming soon!")

    if option == "Chatbox":
        # st.header("⌨️ Text Generation Chatbox")
        st.markdown(
            """
            <h2 class="underline">⌨️ Text Generation Chatbox</h2>
            """,
            unsafe_allow_html=True
        )
        st.write("""
            ###### A standard chatbox that is able to answer your questions.
            ###### Note: The greater the information you provide in your query, the better Kyrax can respond. As Kyrax is still developing and learning day by day, it can make mistakes. Check necessary details.
            """)

        form = st.form(key="user_settings")
        with form:
            st.write("Enter your query below [Example: Startup Idea, Subject of Topic, Name Suggestions]")
            
            query_input = st.text_input("Your Message to Kyrax:", key="message_input")

            st.write("Please select the number of responses you want to generate and the level of creativity")

            col1, col2 = st.columns(2)
            with col1:
                num_input = st.slider(
                    "Number of results",
                    value=3,
                    key="num_input",
                    min_value=1,
                    max_value=10,
                    help="Choose to generate between 1 to 10 results. Greater responses may result in repeated outcomes."
                )
            with col2:
                creativity_input = st.slider(
                    "Creativity", value=0.5,
                    key="creativity_input",
                    min_value=0.1,
                    max_value=0.9,
                    help="Lower values generate more predictable output, higher values generate more creative output."
                )
            generate_button = form.form_submit_button("Generate Result")

            if generate_button:
                if query_input == "":
                    st.error("Input field cannot be blank")
                else:
                    my_bar = st.progress(0.05)
                    st.subheader("Results:")

                    for i in range(num_input):
                        st.markdown("""---""")
                        user_message = text_generation(query_input, creativity_input)
                        st.write(user_message)
                        my_bar.progress((i + 1) / num_input)

    elif option == "Decision Tree Classification":
        # st.header("🌳 Decision Tree Classifier")
        st.markdown(
            """
            <h2 class="underline"> 🌳 Decision Tree Classifier</h2>
            """,
            unsafe_allow_html=True
        )
        st.write("Making decision trees all by yourself can be tiring. Why not let Kyrax do the work while you sit back? 😉")
        st.write("It will automatically fix your data if it has non-numerical values, and show you the original data, as well as the updated data.")
        # st.write("Oh and worried about fixing the data, such as NaN/None/Non-numerical Values? Relax, it will do it for you.")
        uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Original Data:")
            st.write(df)
            
            st.write("Converted Data:")
            df_processed = auto_convert(df)
            st.write(df_processed)

            decision_tree_classifier(df_processed)

    # Footer Section
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(to right, #5D3FD3, #702963); /* Gradient background */
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #BDB5D5;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        .footer img {
            vertical-align: middle;
            height: 20px;
        }
        /* Underline header styles */
        .underline {
            text-decoration: underline;
        }
        </style>
        <div class="footer">
            Created by Abdul Rafay Ahsan
            <img src="https://image.similarpng.com/very-thumbnail/2021/12/Python-programming-logo-on-transparent-background-PNG.png" alt="Python Logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub Logo">
            <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="Streamlit Logo">
            <img src="https://www.datanami.com/wp-content/uploads/2023/06/Cohere-Color-Logo.png" alt="Cohere Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
# Color codes used previously:
#footer = f1f1f1 
# #ff7e5f, #feb47b
# Purple mix = #5D3FD3, #702963
#915F6D

#text = #555

    # elif option == "Classifiers for Prediction":
    #     st.title("🔮 Predicting Certain Data Points")

    #     # Add code for data prediction here
    #     st.write("This page will handle data prediction using your selected model.")

    #     # Example: Input data for prediction, model selection, etc.
    #     st.text_input("Enter data for prediction")
    
    #     # Example: Handle input, make predictions, display results

if __name__ == "__main__":
    main()

