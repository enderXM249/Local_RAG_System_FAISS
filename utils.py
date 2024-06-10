from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
import os
import warnings


# Set up Google Generative AI API key
genai.configure(api_key="AIzaSyCR8MnI9MejtOYsSMljNbBqlZRHWfbxWl0")
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCR8MnI9MejtOYsSMljNbBqlZRHWfbxWl0'

my_llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)

def query_refiner(conversation, query):
    my_prompt = PromptTemplate.from_template(
            "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
        )
    chain = LLMChain(
            llm=my_llm,
            prompt=my_prompt
        )
    input_data = {
        'conversation': conversation,
        'query': query
    }
    response = chain.invoke(input=input_data)
    return response["text"]





# # def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
# #     return response['choices'][0]['text']