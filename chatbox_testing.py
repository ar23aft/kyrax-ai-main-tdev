import cohere
import streamlit as st
# import os
# import textwrap
# import json
# import numpy as np

# Set up Cohere client
co = cohere.Client("wRb0iELnAbGjPMMA0fxkMk3YSdU1MApV5SlG5Z4q") # Get your API key: https://dashboard.cohere.com/api-keys

def text_generation(user_message, temperature):
    """
    This chatbox is for general use. Ask anything.
    Arguments:
        user_message(str): the query or request
        temperature(str): the Generate model 'temperature' value
    Returns:
        response(str): the response to query or request(s)
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
    preamble ="""## Task & Context
    Generate concise responses, with minimum one-sentence. 
    
    ## Style guide
    Be professional.
    """

    # Call the Cohere Chat Endpoint
    # response = co.chat(  # co.chat seems outdated, hence use co.generate
    response = co.generate(
        model = 'command-r',
        user_message = prompt + preamble,
        temperature = temperature
        max_tokens = 150 # to be used with co.generate
    )
        # chat_history = response.chat_history

    return response.generate[text]

    # Generate the response, along with current chat history
    # response = co.chat(message=message,
    #                    preamble=preamble,
    #                    chat_history=response.chat_history)

    # View the chat history:
    # for turn in response.chat_history:
    #     print("Role:",turn.role)
    #     print("Message:",turn.message,"\n")
    
    # return response.text
    # print(response.text)



# The FRONT END Code starts here:

st.title("🚀 Kyrax A.I")


form = st.form(key="user_settings")
with form:
  st.write("Enter your query below [Example: Startup Idea, Introduction Message, Name Suggestions] ")
  # User input - Industry name
  query_input = st.text_input("User Message", key = "message_input")

  # Create a two-column view
  col1, col2 = st.columns(2)
  with col1:
      # User input - The number of ideas to generate
      num_input = st.slider(
        "Number of results", 
        value = 3, 
        key = "num_input", 
        min_value=1, 
        max_value=10,
        help="Choose to generate between 1 to 10 results")
  with col2:
      # User input - The 'temperature' value representing the level of creativity
      creativity_input = st.slider(
        "Creativity", value = 0.5, 
        key = "creativity_input", 
        min_value=0.1, 
        max_value=0.9,
        help="Lower values generate more “predictable” output, higher values generate more “creative” output")  
  # Submit button to start generating ideas
  generate_button = form.form_submit_button("Generate Result")

  if generate_button:
    if query_input == "":
        st.error("Input/Query field cannot be blank")
    else:
        my_bar = st.progress(0.05)
        st.subheader("Results:")

        for i in range(num_input):
            st.markdown("""---""")
            response_text = text_generation(query_input, creativity_input)
            # response_to_query = text_generation(user_message,creativity_input)
            # st.markdown("##### " + response_to_query)
            st.write(response_text)
            my_bar.progress((i+1)/num_input)
