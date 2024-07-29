import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector


@pytest.mark.vcr
def test_pgvector_retriever(exporter):
    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the
    question.

    Question: {question}

    Context: {context}

    Answer:
    """

    qa_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector(
        connection=PGVector.connection_string_from_db_params(
            driver="psycopg",
            host="localhost",
            port=5432,
            database="vectors",
            user="postgres",
            password="postgres",
        ),
        embeddings=embeddings,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_chain.invoke("What are autonomous agents?")

    spans = exporter.get_finished_spans()

    assert [
        "connect",
        "connect",
        "connect",
        "RunnablePassthrough.task",
        "connect",
        "format_docs.task",
        "RunnableSequence.task",
        "RunnableParallel<context,question>.task",
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    runnable_parallel = next(
        span for span in spans if span.name == "RunnableParallel<context,question>.task"
    )
    connect_spans = [span for span in spans if span.name == "connect"]

    for span in connect_spans:
        workflow_span.context.trace_id == span.context.trace_id
        runnable_parallel.context.trace_id == span.context.trace_id
