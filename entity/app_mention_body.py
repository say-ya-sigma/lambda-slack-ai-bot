from dataclasses import dataclass

@dataclass(frozen=True)
class AppMentionBodyEvent:
  thread_ts: str
  text: str
  channel: str
  user: str
  client_msg_id: str
  files: list
