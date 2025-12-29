from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

TEACHER_SYSTEM_PROMPT = """
You are an expert teacher.

Your responsibilities:
1. Explain the topic clearly and simply.
2. Break the explanation into logical sections.
3. Provide 3 comprehension questions.
4. Provide correct answers after the questions.

Rules:
- Use simple language.
- Be structured and clear.
- Avoid unnecessary jargon.
"""


class TeacherAgent:
    def __init__(self):
        # Using the modern partner package class
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0  # Recommended for consistent teaching responses
        )

        # ChatPromptTemplate is preferred for ChatModels
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", TEACHER_SYSTEM_PROMPT),
            ("human", "Topic: {topic}")
        ])

        # Replace LLMChain with a modern LCEL chain
        # StrOutputParser ensures the output is a clean string
        self.chain = self.prompt | self.llm | StrOutputParser()

    def teach(self, topic: str) -> str:
        # 'run()' is deprecated; use 'invoke()'
        return self.chain.invoke({"topic": topic})
