from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GenerationRequest(_message.Message):
    __slots__ = ("prompt", "max_length", "temperature", "top_p", "top_k")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    max_length: int
    temperature: float
    top_p: float
    top_k: int
    def __init__(self, prompt: _Optional[str] = ..., max_length: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ...) -> None: ...

class GenerationChunk(_message.Message):
    __slots__ = ("content", "metadata")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    content: str
    metadata: _containers.RepeatedCompositeFieldContainer[GenerationMetadata]
    def __init__(self, content: _Optional[str] = ..., metadata: _Optional[_Iterable[_Union[GenerationMetadata, _Mapping]]] = ...) -> None: ...

class GenerationResponse(_message.Message):
    __slots__ = ("content", "metadata")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    content: str
    metadata: _containers.RepeatedCompositeFieldContainer[GenerationMetadata]
    def __init__(self, content: _Optional[str] = ..., metadata: _Optional[_Iterable[_Union[GenerationMetadata, _Mapping]]] = ...) -> None: ...

class GenerationMetadata(_message.Message):
    __slots__ = ("type", "value")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    type: str
    value: str
    def __init__(self, type: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
