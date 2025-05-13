import transformers


def test_placeholder():
    pipeline = transformers.pipeline(
        task="text-generation", model="gpt2", framework="tf"
    )
    response = pipeline("the secret to baking a really good cake is ")
    print(type(response))
    print(response[0])
    print(type(response[0]))
    raise AssertionError
