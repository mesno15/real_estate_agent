
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from utils import create_db, get_llm, get_history

OPENAI_API_KEY=''


def sim_search(query, db, llm):
    similar_docs = db.similarity_search(query, k=3)
    print(f"**Similar documents: **\n{similar_docs}")
    prompt = PromptTemplate(
        template="{query}\nContext: {context}",
        input_variables=["query", "context"],
    )
    chain = load_qa_chain(llm, prompt = prompt, chain_type="stuff")
    print(
        f"**Recommendation based on similarity search:** \n"\
        f"{chain.run(input_documents=similar_docs, query = query)}")
    return similar_docs


def augment_search(query, db, llm, history):
    chain = ConversationalRetrievalChain.from_llm(
        llm, 
        db.as_retriever()
    )
    response = chain({"question": query, "chat_history": history})
    print(
        f"**Recommendation based on buyer's preferences: **\n"\
        f"{response['answer']}")


def main():
    file_path = "./data_generated.csv"
    db = create_db(file_path, OPENAI_API_KEY)
    llm = get_llm()

    query_for_similarity_search = """
        Based on the listings in the context recommend me a home if 
        I want to live near the sea and I have unlimited budget.
        """
    sim_search(query_for_similarity_search, db, llm)

    personal_questions = [   
                "How big do you want your house to be?" ,
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
            ]
    personal_answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]
    history = get_history(personal_questions, personal_answers)

    query = """
        Choose one home listing recommended for the user out of the given documents. 
        Try to convince the buyer, why this home would be suitable for them, include all of the previous answers in your explanation.
        Do not change any factual information of the properties.
        """
    augment_search(query, db, llm, history)  


if __name__ == "__main__":
    main()
    
    
