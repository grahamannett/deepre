from datetime import datetime

import reflex as rx
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from deepre.pages.index import index
from deepre.prompts import message_template
from deepre.prompts.message_template import MessageTemplate
from deepre.provider import LLMProvider
from deepre.state import ResearchState
from deepre.app_config import conf
from deepre import query

# Create app
app = rx.App()
app.add_page(index, title="Research Assistant")
