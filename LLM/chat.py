class Chat:
    """
    Handles the chat to avoid OOM (Out of Memory) issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        self.buffer = []

    def append(self, item):
        """
        Appends a new message to the buffer.
        If the number of messages in the buffer exceeds the set size, 
        the oldest two messages will be removed to control memory usage.
        """
        self.buffer.append(item)
        if len(self.buffer) > 2 * self.size:
            # Each time a user message and an assistant message are added,
            # the oldest two messages are removed.
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        """
        Initializes the chat by setting the initial system message.
        """
        self.init_chat_message = init_chat_message

    def to_list(self):
        """
        Returns the current chat history, including the initial system message (if it exists).
        """
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer

