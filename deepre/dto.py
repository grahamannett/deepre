from pydantic import BaseModel

"""Not sure if ill use these but are temporarily here for reference on the SERP related agent"""


class SerpOrganicResult(BaseModel):
    position: int
    title: str
    link: str
    redirect_link: str
    displayed_link: str
    favicon: str
    date: str
    snippet: str
    snippet_highlighted_words: list[str]
    source: str


class SerpResponse(BaseModel):
    search_metadata: dict
    search_parameters: dict
    search_information: dict
    knowledge_graph: dict
    inline_images: list
    inline_videos: list
    related_questions: list
    organic_results: list[SerpOrganicResult]
    top_stories: list
    top_stories_link: str
    top_stories_serpapi_link: str
    related_searches: list
    pagination: dict
    serpapi_pagination: dict
