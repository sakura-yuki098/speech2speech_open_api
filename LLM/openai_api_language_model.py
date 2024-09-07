from openai import OpenAI
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
import time

logger = logging.getLogger(__name__)
console = Console()

class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url=None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你的名字叫花火，你是用户的朋友，始终以中文回答，并提供帮助。回答字数限制在3句话以内，我说继续，你可以继续回答。注意，你现在运行并生成的文字，是会被转换为语音输出的。请用自然的方式说话。"},
                {"role": "user", "content": "Hello"},
            ],
            stream=self.stream
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}: warmed up! time: {(end - start):.3f} s"
        )

    def save_to_file(self, text):
        with open('debug_log.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def process(self, prompt):
        logger.debug("call api language model...")
        
        # Ensure prompt is a string
        if isinstance(prompt, tuple):
            prompt = prompt[0]  # If it's a tuple, take the first element as the string
        
        # Append user input to the chat context
        self.chat.append({"role": self.user_role, "content": prompt})
        
        # Build the complete message context, including history and system prompts
        messages = self.chat.to_list()

        # Check if the messages list format is correct
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                self.save_to_file(f"Invalid message format: {message}")
                raise ValueError(f"Invalid message format: {message}")

        # Call the API to generate a response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=self.stream
        )
        
        # Get the generated text
        generated_text = response.choices[0].message.content
        
        # Append the generated response to the chat context
        self.chat.append({"role": "assistant", "content": generated_text})
        
        # Output the generated text and save it to a file
        self.save_to_file(f"Generated text: {generated_text}")
        yield generated_text


