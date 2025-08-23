from typing import Callable


class Config:
    exception_logger = None
    use_legacy_attributes = True
    upload_base64_image: Callable[[str, str, str, str], str] = None
