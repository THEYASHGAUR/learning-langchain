from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.core.prompts import PromptTemplate
from langchain.core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)

chain = template | model | StrOutputParser()    

result = chain.invoke({"country":"India"})

print(result)