from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY='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'


def main():
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, 
        model_name="gpt-3.5-turbo", 
        temperature=0
    )
    instruction = """
        You are generating real estate listing data.
        An example of a listing:
        
        Neighborhood: Green Oaks
        Price: $800,000
        Bedrooms: 3
        Bathrooms: 2
        House Size: 2,000 sqft

        Description: Welcome to this eco-friendly oasis nestled in the 
        heart of Green Oaks. This charming 3-bedroom, 2-bathroom home 
        boasts energy-efficient features such as solar panels and a 
        well-insulated structure. Natural light floods the living 
        spaces, highlighting the beautiful hardwood floors and 
        eco-conscious finishes. The open-concept kitchen and dining 
        area lead to a spacious backyard with a vegetable garden, 
        perfect for the eco-conscious family. Embrace sustainable 
        living without compromising on style in this Green Oaks gem.

        Neighborhood Description: Green Oaks is a close-knit, 
        environmentally-conscious community with access to organic 
        grocery stores, community gardens, and bike paths. 
        Take a stroll through the nearby Green Oaks Park or grab a cup 
        of coffee at the cozy Green Bean Cafe. With easy access to 
        public transportation and bike lanes, commuting is a breeze.

            
        Task:  
          
        Generate  26 real estate listings, produce descriptions 
        of various properties. 
        Write the answer in a csv format, all strings should be double quoted.
        Attributes should also be double quouted in the first row.


    """

    resp = llm.invoke(instruction)
    with open("data_generated.csv", "w") as f:
        f.write(resp.content)

    
if __name__ == "__main__":
    main()

