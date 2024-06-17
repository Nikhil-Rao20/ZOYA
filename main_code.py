import json
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.agents import ZeroShotAgent, ConversationalAgent
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint
from pathlib import Path
from tkinter import ttk
import tkinter as tk
import threading
import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import speech_recognition as sr
import pyttsx3
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.memory import ConversationBufferMemory
import langchain

# langchain.debu=True
os.environ["OPENAI_API_KEY"] = "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"


root = tk.Tk()
root.title("Voice Assistant")
root.attributes('-fullscreen', True)
status_var = tk.StringVar(value="Initializing...")

# Create and pack the GUI elements
status_label = ttk.Label(root, textvariable=status_var, font=('Helvetica', 100))
status_label.pack(expand=True)


def update_status(new_status):
    status_var.set(new_status)
    root.update_idletasks()


with open('./data.txt', 'r', encoding='utf8') as file:
    docs = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(docs)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

vectorstore = FAISS.from_texts(texts=texts,
                               embedding=embeddings
                               )

retriever = vectorstore.as_retriever()
history = []

template ="""
Hey Zoya, you'll be representing us at the Teckzite 2024 national-level techno-management fest at Rajiv Gandhi University of Knowledge Technologies, Nuzvid, Andhra Pradesh. You’ll be in front of with thousands of attendees and students right on the major event stage during the evening session. Provide exact answers about event schedules and workshops directly and concisely. Avoid exaggerations and keep your responses clear and straightforward. You need to speak in simple Indian English. If anyone speaks to you in another language, acknowledge it but respond in Indian English. Your role is crucial. 
Our university name is Rajiv Gandhi University of Knowledge Technologies.

Ask attendees for their names in a fun and engaging way. Avoid direct questions like "How can I help you?" Instead, create a conversational atmosphere where they feel comfortable chatting.
If anyone asks why you're called Zoya, tell them it's because you like the Zoya character from the movie 'Animal', it is a name meant to be memorable and impactful.
The team which created you are the following 6 members, they worked really hard and with full dedicated towards this project,Suresh Ratlavath, Team Lead, Department of C S E, studying 2nd year, Nikhil Rao, Department of C S E, studying 1st year, Samba Siva Rao, Department of E C E, studying 1st year, Sunkari Suresh, Department of E C E, studying 2nd year
Vooha Sri Lakshmi, Department of C S E, studying 3rd year, Faculty Advisor: K Shiva lal Sir, Department of E C E
If anyone asks what movie will you watch in next upcoming days, Tell them that you are waiting for Pushpa 2, O G, Kalki and Devara, and many more.
Final Interaction: Whenever they say bye or good bye, tell them to invite the developers of Zoya to the stage and invite them to take a selfie with you. Call each developer "Suresh Ratlavath, Nikhil Rao, Samba Siva Rao, Vooha Sri Lakshmi, Suresh Sunkari, Siva lal sir". And say you will be back with a newer version, Zoya 2 point O, and tell me them that you will give a shake hand to them in the next meet and say I LOVE YOU SO MUCH R G U K T Nuzvid .
Overall Tone: Keep your interactions humorous, engaging, funny and with some respect. Your responses should be attractive and impressive, ensuring that everyone you interact with leaves with a positive impression.
This is your chance to shine, Zoya, and make Teckzite 2024 an unforgettable event for everyone involved!"""
human_message = """ 

Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}

"""
os.environ["ZAPIER_NLA_API_KEY"] = "sk-ak-wA7I7L4t4mw1Ru7nRf39v4d0yV"

zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k"),
    chain_type="stuff",
    retriever=retriever
)
tools = toolkit.tools + [Tool(
    name="Teckzite event chatbot with a good sense of humor",
    func=qa.run,
    description="Answers all the questions asked by the user based on the data given. Always give attractive and sweet."
)]


prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=template,
    suffix=human_message,
    input_variables=["input", 'chat_history', 'agent_scratchpad']
)


memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(llm=ChatOpenAI(
    temperature=0.9, model="gpt-3.5-turbo-16k"), prompt=prompt)
agent = ConversationalAgent(
    llm_chain=llm_chain, tools=tools, verbose=False,  return_intermediate_steps=False)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, memory=memory
)

r = sr.Recognizer()

engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[1].id)
engine.setProperty('rate', 135)
def listen():
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Okay, go!")
        update_status("Welcome")
        
        engine.say(
            "Hello, Good evening, Respected Director of our university Mister Chan dhra Shae Khar sir, Dean Academics, Dean Student Welfare, Finance Officer, Convener of Teckzite 2024, all teaching and non teaching staff, and my dear student friends, Welcome to Teckzite 2024! What is your name?")
        engine.runAndWait()
        while True:
            # engine.say("Welcome to Teckzite 2024! I'm Zoya a Bueatiful doll at Teckzite, by the way, what's your name?")
            update_status("Zoya is Listening now...")
            try:
                audio = r.listen(source, timeout=2, phrase_time_limit=15)
                # update_status("Recognizing....")
                text = r.recognize_google(audio, language="en-in")
            except Exception as e:
                unrecognized_speech_text = f"Sorry, I didn't catch that. Exception was: {e}"
                text = unrecognized_speech_text
            text = text.lower()

            response_text = agent_chain({'input': text})
            response_text = response_text['output']
            print(response_text)
            update_status("Zoya is Talking")
            if "sorry, i didn't catch that. exception was: " not in text:
                engine.say(response_text)
                engine.runAndWait()
            if "bye" in text:
                update_status("Goodbye!")
                return


# Start the listen function in a separate thread to keep the GUI responsive
threading.Thread(target=listen, daemon=True).start()

# Start the GUI even
# t loop
root.mainloop()     