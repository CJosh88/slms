"""Single prompt template shared by every model so the comparison is fair."""

SYSTEM = (
    "You are a contract review assistant. Read the contract excerpt and answer the "
    "question with exactly one word: Yes or No."
)

USER_TEMPLATE = """Contract excerpt:
<<<
{context}
>>>

Question: {question}"""


def build_user(context: str, question: str) -> str:
    return USER_TEMPLATE.format(context=context, question=question)
