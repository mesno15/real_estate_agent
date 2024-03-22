import chromadb
from langchain.chains import RetrievalQA, ConversationChain, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, CombinedMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import Any, Dict, Optional, Tuple
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY=''


class MementoBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_ai_message(output_str)


def create_db(file_path, openai_api_key):
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": [
                "Neighborhood", 
                "Price",
                "Bedrooms",
                "Bathrooms",
                "House Size",
                "Description",
                "Neighborhood Description"
            ],
        }
    )
    data = loader.load()
    cts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = cts.split_documents(data)
    emb = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )
    db = Chroma.from_documents(split_docs, emb)

    return db


def get_llm():
    model_name = 'gpt-3.5-turbo'
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name, 
        temperature=0, 
        max_tokens=2000, 
    )
    return llm


def get_history(questions, answers):
    history = ChatMessageHistory()
    history.add_user_message(
        f"""
        You are AI that will enrich the desrciptions for home 
        listings for a user based on their answers to personal 
        questions.The augmentation should personalize the listing 
        without changing factual information. 
        Ask user {len(questions)} questions
        """
    )
    for i in range(len(questions)):
        history.add_ai_message(questions[i])
        history.add_user_message(answers[i])

    history_tuples = []
    for i in range(0, int(len(history.messages)), 2):
        try:
            history_tuples.append((history.messages[i].content, history.messages[i+1].content))
        except IndexError:
            history_tuples.append((history.messages[i].content, ""))
    #print(history_tuples)
    return history_tuples
