from __future__ import annotations


def search_web(query: str, api_key: str | None, max_results: int = 5) -> list[dict[str, str]]:
    if not api_key:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
        return [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            }
            for result in response.get("results", [])
        ]
    except Exception as exc:
        return [{"title": "Web search unavailable", "url": "", "content": str(exc)}]
