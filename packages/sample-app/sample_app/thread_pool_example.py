import pinecone
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from concurrent.futures import ThreadPoolExecutor
import contextvars
import functools

Traceloop.init("thread_pool_example")


@workflow("retrieval_flow")
def do_retrieval(index: pinecone.Index):
    with ThreadPoolExecutor(max_workers=3) as executor:
        for _ in range(3):
            # Note: this is needed instead of calling `submit` directly, like this:
            # executor.submit(index.query, [1.0, 2.0, 3.0], top_k=10)
            ctx = contextvars.copy_context()
            executor.submit(
                ctx.run,
                functools.partial(index.query, [1.0, 2.0, 3.0], top_k=10),
            )


def get_index():
    INDEX_NAME = "thread-pool-repro"
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=3, metric="dotproduct")
    return pinecone.Index(INDEX_NAME)


def main():
    index = get_index()
    do_retrieval(index)


if __name__ == "__main__":
    main()
