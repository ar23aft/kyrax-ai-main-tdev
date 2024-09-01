import cohere
import streamlit as st

# Set up Cohere client
co = cohere.Client("wRb0iELnAbGjPMMA0fxkMk3YSdU1MApV5SlG5Z4q")  # Replace with your actual API key

def text_generation(user_message, temperature: float, chat_history=None):
    """
    This chatbox is for general use. Ask anything.
    Arguments:
        user_message(str): the query or request
        temperature(float): the chat model's 'temperature' value
        chat_history(list): optional; the previous chat history
    Returns:
        response(str): the response to query or request(s)
        chat_history(list): the updated chat history
    """

    # Create a custom prompt
    prompt = f"""
    ## Task & Context
    Generate concise responses, with minimum one-sentence.
    
    ## Style guide
    Be professional.
    
    User Message: {user_message}
    """

    # Call the Cohere Generate Endpoint
    response = co.generate(
        model='command',  # Replace with a valid model name if different
        prompt=prompt,
        temperature=temperature,
        max_tokens=150
    )

    # Get the response text
    response_text = response.generations[0].text.strip() if response.generations else "No response generated."
    
    # Update chat history if needed
    updated_chat_history = chat_history  # Adjust as necessary if chat history is maintained

    return response_text, updated_chat_history


# The FRONT END Code starts here:

st.title("🚀 Kyrax A.I 🤖")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

form = st.form(key="user_settings")
with form:
    st.write("Enter your query below [Example: Startup Idea, Introduction Message, Name Suggestions]")
    # User input - Industry name
    query_input = st.text_input("User Message", key="message_input")

    # Create a two-column view
    col1, col2 = st.columns(2)
    with col1:
        # User input - The number of ideas to generate
        num_input = st.slider(
            "Number of results",
            value=3,
            key="num_input",
            min_value=1,
            max_value=10,
            help="Choose to generate between 1 to 10 results")
    with col2:
        # User input - The 'temperature' value representing the level of creativity
        creativity_input = st.slider(
            "Creativity", value=0.5,
            key="creativity_input",
            min_value=0.1,
            max_value=0.9,
            help="Lower values generate more predictable output, higher values generate more creative output")
    # Submit button to start generating ideas
    generate_button = form.form_submit_button("Generate Result")

    if generate_button:
        if query_input == "":
            st.error("Input/Query field cannot be blank")
        else:
            my_bar = st.progress(0.05)
            st.subheader("Results:")

            # Iterate to generate multiple responses if needed
            for i in range(num_input):
                st.markdown("""---""")
                response_text, st.session_state['chat_history'] = text_generation(
                    query_input,
                    creativity_input,
                    st.session_state['chat_history']
                )
                st.write(response_text)
                my_bar.progress((i+1)/num_input)
