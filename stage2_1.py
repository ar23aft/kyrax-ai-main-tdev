import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import cohere

# Initialize Cohere client (for the Chatbox feature)
co = cohere.Client("wRb0iELnAbGjPMMA0fxkMk3YSdU1MApV5SlG5Z4q")  # Replace with your actual API key

# Dropdown for selecting the feature
st.sidebar.title("Select a Function")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Chatbox", "DecisionTree Classification", "Prediction"])

if app_mode == "Chatbox":
    st.title("🚀 Kyrax A.I Chatbox")

    def text_generation(user_message, temperature: float, chat_history=None):
        # Your chatbox logic here
        prompt = f"""
        ## Task & Context
        Generate concise responses, with minimum one-sentence.

        ## Style guide
        Be professional.

        User Message: {user_message}
        """
        response = co.generate(
            model='command',  # Replace with a valid model name if different
            prompt=prompt,
            temperature=temperature,
            max_tokens=150
        )

        response_text = response.generations[0].text.strip() if response.generations else "No response generated."
        updated_chat_history = chat_history

        return response_text, updated_chat_history

    # Chatbox input and output logic
    st.write("Enter your query below [Example: Startup Idea, Introduction Message, Name Suggestions] ")
    query_input = st.text_input("User Message", key="message_input")
    # Handle input and display chatbox results

elif app_mode == "DecisionTree Classification":
    st.title("🌳 DecisionTree Classification")

    # File uploader for the dataset
    uploaded_file = st.file_uploader("Choose a dataset (CSV file)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.write(df)

        # Replace NaN or None values with 0
        df.fillna(0, inplace=True)

        # Convert non-numerical columns to numerical values automatically
        df_encoded = df.copy()
        for column in df_encoded.columns:
            if df_encoded[column].dtype == 'object':
                df_encoded[column] = df_encoded[column].astype('category').cat.codes
        
        st.write("Data after Encoding Non-Numerical Columns:")
        st.write(df_encoded)

        # Feature and target selection
        features = df_encoded.columns[:-1].tolist()  # All columns except the last one
        X = df_encoded[features]
        y = df_encoded[df_encoded.columns[-1]]

        # Train the Decision Tree classifier
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X, y)

        # Plot and display the Decision Tree
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a figure and axis for the plot
        tree.plot_tree(dtree, feature_names=features, filled=True, ax=ax)
        st.pyplot(fig)  # Display the plot in Streamlit

elif app_mode == "Prediction":
    st.title("🔮 Prediction")

    # Add code for data prediction here
    st.write("This page will handle data prediction using your selected model.")

    # Example: Input data for prediction, model selection, etc.
    st.text_input("Enter data for prediction")

    # Example: Handle input, make predictions, display results
