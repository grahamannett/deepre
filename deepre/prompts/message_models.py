from pydantic import BaseModel


class FollowUpQuery(BaseModel):
    query: str
