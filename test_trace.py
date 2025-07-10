from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.tracers.langchain import wait_for_all_tracers


@traceable
def hello():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return llm.invoke("Say hi to LangSmith!").content


print("Result:", hello())
wait_for_all_tracers()
print("Trace flushed.")
