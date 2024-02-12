import pytest
import replicate


@pytest.mark.vcr
def test_replicate_image_generation(exporter):
    replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": "robots"},
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.run",
    ]


@pytest.mark.vcr
def test_replicate_image_generation_predictions(exporter):
    model = replicate.models.get("kvfrans/clipdraw")
    version = model.versions.get(
        "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b"
    )
    replicate.predictions.create(version, input={"prompt": "robots"})

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.predictions.create",
    ]
