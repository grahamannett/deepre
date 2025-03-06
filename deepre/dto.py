from functools import cache
from typing import Any

from pydantic import BaseModel, TypeAdapter, computed_field

from deepre.utils.cache_utils import create_url_hash


@cache
def _get_ta(type_: type) -> TypeAdapter:
    """
    Use this to allow for dynamic type validation with cache
    """
    return TypeAdapter(type_)


class Failed(BaseModel):
    """Failed to generate queries."""

    ...


class PageResult(BaseModel):
    """Result of a web page search."""

    url: str
    title: str
    date: str = ""
    description: str | None = None
    page_text: str | None = None
    hash: str | None = None  # Will be populated during validation

    def model_post_init(self, __context: Any) -> None:
        """Calculate the hash if not already set."""
        if not self.hash:
            self.hash = create_url_hash(self.url, date=self.date)


class SerpOrganicResult(BaseModel):
    position: int
    title: str
    link: str
    displayed_link: str
    source: str
    favicon: str
    redirect_link: str | None = None
    date: str | None = None
    snippet: str | None = None
    snippet_highlighted_words: list[str] | None = None

    @computed_field
    @property
    def hash(self) -> str:
        return create_url_hash(self.link, date=self.date or "")

    @classmethod
    def ta_response_validate_python(cls, results: dict) -> list["SerpOrganicResult"]:
        return _get_ta(list[cls]).validate_python(results.get("organic_results", results))


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


# type adapters
# -- better to have this explicit than to use some dynamic way
serp_organic_results_ta = TypeAdapter(list[SerpOrganicResult])
