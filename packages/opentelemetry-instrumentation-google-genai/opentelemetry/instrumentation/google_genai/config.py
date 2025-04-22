from typing import Callable, Coroutine, Optional


class Config:
    exception_logger = None
    upload_base64_image: Optional[
        Callable[[str, str, str, str], Coroutine[None, None, str]]
    ] = None
    convert_image_to_openai_format: bool = True
