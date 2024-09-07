class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        self.buffer = []

    def append(self, item):
        """
        将新的消息追加到 buffer 中。
        如果 buffer 中的消息数超过了设定的 size，最早的两条消息将被移除，以控制内存使用。
        """
        self.buffer.append(item)
        if len(self.buffer) > 2 * self.size:
            # 每次添加一个用户消息和一个助手消息，所以移除最早的两条
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        """
        初始化聊天，设置初始的系统消息。
        """
        self.init_chat_message = init_chat_message

    def to_list(self):
        """
        返回当前的聊天记录，包括初始化的系统消息（如果存在）。
        """
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer
