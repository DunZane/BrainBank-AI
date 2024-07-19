import grpc
from typing import Any, Iterator, List, Mapping, Optional

from internal.rpc import message_pb2
from internal.rpc import message_pb2_grpc

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from langchain.llms.base import LLM


class Chatglm6b(LLM):
    base_url: str
    temperature: float
    streaming: bool
    top_k: int
    top_p: float
    max_length: int

    def __init__(self,
                 base_url: str,
                 temperature: float,
                 streaming: bool,
                 top_k: int,
                 top_p: float,
                 max_length: int):
        super().__init__()
        self.base_url = base_url
        self.temperature = temperature
        self.streaming = streaming
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length

    @property
    def _llm_type(self) -> str:
        return 'chatglm-6b'

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if stop is None:
            raise ValueError("stop sequences are not supported.")

        channel = grpc.insecure_channel(self.base_url)
        stub = message_pb2_grpc.GenerationServiceStub(channel)

        message = message_pb2.GenerationRequest(
            prompt=prompt,
            temperature=self.temperature,
            max_length=self.max_length,
            top_p=self.top_p,
            top_k=self.top_k
        )
        response = stub.Generation(message)

        return response.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "base_url": self.base_url,
            "temperature": self.temperature,
            "streaming": self.streaming,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_length": self.max_length
        }

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if stop is None:
            raise ValueError("stop sequences are not supported.")

        channel = grpc.insecure_channel(self.base_url)
        stub = message_pb2_grpc.GenerationServiceStub(channel)

        message = message_pb2.GenerationRequest(
            prompt=prompt,
            temperature=self.temperature,
            max_length=self.max_length,
            top_p=self.top_p,
            top_k=self.top_k
        )
        response_stream = stub.GenerationStream(message)

        for response in response_stream:
            yield GenerationChunk(content=response.content)
