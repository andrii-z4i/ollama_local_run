import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.image_embedding_manager import ImageEmbeddingManager

# Global embedding manager — initialized on server startup
_embedding: Optional[ImageEmbeddingManager] = None

mcp_app = FastMCP(
    "photo-memory",
    instructions=(
        "A personal photo memory system. Search and explore your photo library "
        "using natural language. Photos have been analyzed by a vision model and "
        "stored with metadata including dates, locations, and descriptions."
    ),
)


def _get_embedding() -> ImageEmbeddingManager:
    if _embedding is None:
        raise RuntimeError("Embedding manager not initialized")
    return _embedding


def _docs_to_results(docs) -> str:
    """Format search results as a readable string."""
    if not docs:
        return "No photos found matching your query."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts = [f"**Photo {i}**"]
        if meta.get("file_name"):
            parts.append(f"  File: {meta['file_name']}")
        if meta.get("date_taken"):
            parts.append(f"  Date: {meta['date_taken']}")
        if meta.get("location"):
            parts.append(f"  Location: {meta['location']}")
        if meta.get("source"):
            parts.append(f"  Path: {meta['source']}")
        parts.append(f"  Description: {doc.page_content[:500]}")
        results.append("\n".join(parts))

    return "\n\n".join(results)


@mcp_app.tool()
def search_photos(query: str, limit: int = 10) -> str:
    """Search photos by natural language description.

    Use this to find photos matching a description, activity, scene, or object.
    Examples: 'sunset on a beach', 'birthday party', 'group photo with friends'

    Args:
        query: Natural language search query
        limit: Maximum number of results (default 10)
    """
    emb = _get_embedding()
    docs = emb.search_photos(query, limit=limit)
    return _docs_to_results(docs)


@mcp_app.tool()
def search_photos_by_date(
    date: str = "",
    start_date: str = "",
    end_date: str = "",
    query: str = "",
    limit: int = 20,
) -> str:
    """Search photos by date or date range, with optional text query.

    Dates should be in ISO format (YYYY-MM-DD). Provide either 'date' for a
    single day, or 'start_date'/'end_date' for a range.

    Args:
        date: Single date (YYYY-MM-DD) to search
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)
        query: Optional text query to combine with date filter
        limit: Maximum number of results (default 20)
    """
    emb = _get_embedding()
    all_meta = emb.get_all_metadata()

    # Filter by date
    matched = []
    for entry in all_meta:
        dt = entry.get("date_taken", "")
        if not dt:
            continue
        date_part = dt[:10]  # Extract YYYY-MM-DD from ISO datetime

        if date and date_part == date:
            matched.append(entry)
        elif start_date and end_date and start_date <= date_part <= end_date:
            matched.append(entry)
        elif start_date and not end_date and date_part >= start_date:
            matched.append(entry)
        elif end_date and not start_date and date_part <= end_date:
            matched.append(entry)

    if not matched:
        return "No photos found for the specified date(s)."

    # If a text query is also provided, do semantic search within matched files
    if query:
        source_paths = [m["source"] for m in matched if m.get("source")]
        # Search semantically and filter to only date-matched files
        docs = emb.search_photos(query, limit=limit * 3)
        docs = [d for d in docs if d.metadata.get("source") in set(source_paths)]
        return _docs_to_results(docs[:limit])

    # Format results sorted by date
    matched.sort(key=lambda x: x.get("date_taken", ""))
    matched = matched[:limit]

    results = []
    for i, entry in enumerate(matched, 1):
        parts = [f"**Photo {i}**"]
        if entry.get("file_name"):
            parts.append(f"  File: {entry['file_name']}")
        parts.append(f"  Date: {entry.get('date_taken', 'unknown')}")
        if entry.get("location"):
            parts.append(f"  Location: {entry['location']}")
        if entry.get("source"):
            parts.append(f"  Path: {entry['source']}")
        if entry.get("description"):
            parts.append(f"  Description: {entry['description'][:500]}")
        results.append("\n".join(parts))

    return "\n\n".join(results)


@mcp_app.tool()
def search_photos_by_location(location: str, limit: int = 10) -> str:
    """Search photos by location name.

    Searches both the location metadata and photo descriptions for location terms.
    Examples: 'Paris', 'Japan', 'Central Park'

    Args:
        location: Location name to search for
        limit: Maximum number of results (default 10)
    """
    emb = _get_embedding()

    # First try semantic search with location context
    docs = emb.search_photos(f"photo taken in {location}", limit=limit * 3)

    # Also check metadata location field
    all_meta = emb.get_all_metadata()
    location_lower = location.lower()
    meta_matches = set()
    for entry in all_meta:
        loc = entry.get("location", "")
        if loc and location_lower in loc.lower():
            meta_matches.add(entry.get("source"))

    # Combine: prioritize metadata matches, then semantic matches
    seen = set()
    combined = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src in meta_matches and src not in seen:
            combined.insert(0, doc)
            seen.add(src)
        elif src not in seen:
            combined.append(doc)
            seen.add(src)

    return _docs_to_results(combined[:limit])


@mcp_app.tool()
def get_photo_details(file_path: str) -> str:
    """Get full details about a specific photo by its file path.

    Args:
        file_path: Full path to the photo file
    """
    emb = _get_embedding()
    docs = emb.search_photos("photo", limit=1, filter_dict={"source": file_path})
    if not docs:
        return f"No photo found at path: {file_path}"

    doc = docs[0]
    meta = doc.metadata
    parts = ["**Photo Details**"]
    for key, value in sorted(meta.items()):
        if key != "checksum":
            parts.append(f"  {key}: {value}")
    parts.append(f"\n**Full Description:**\n{doc.page_content}")
    return "\n".join(parts)


@mcp_app.tool()
def get_photo_timeline(start_date: str, end_date: str) -> str:
    """Get a chronological summary of photos in a date range.

    Returns photos grouped by day for the specified period.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    emb = _get_embedding()
    all_meta = emb.get_all_metadata()

    # Filter and group by day
    by_day = {}
    for entry in all_meta:
        dt = entry.get("date_taken", "")
        if not dt:
            continue
        date_part = dt[:10]
        if start_date <= date_part <= end_date:
            by_day.setdefault(date_part, []).append(entry)

    if not by_day:
        return f"No photos found between {start_date} and {end_date}."

    parts = [f"**Photo Timeline: {start_date} to {end_date}**\n"]
    for day in sorted(by_day.keys()):
        entries = by_day[day]
        locations = set(e.get("location", "") for e in entries if e.get("location"))
        parts.append(f"📅 **{day}** — {len(entries)} photo(s)")
        if locations:
            parts.append(f"   Locations: {', '.join(locations)}")
        for entry in entries[:5]:  # Show up to 5 per day
            desc = entry.get("description", "")[:200]
            name = entry.get("file_name", "unknown")
            parts.append(f"   • {name}: {desc}")
        if len(entries) > 5:
            parts.append(f"   ... and {len(entries) - 5} more")
        parts.append("")

    return "\n".join(parts)


if __name__ == "__main__":
    import sys

    # Parse args manually to avoid conflict with MCP's own arg handling
    chroma_db_name = "photo_db"
    chroma_db_path = "./chroma_db"
    ollama_url = "http://localhost:11434"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--chroma-db-name" and i + 1 < len(args):
            chroma_db_name = args[i + 1]
            i += 2
        elif args[i] == "--chroma-db-path" and i + 1 < len(args):
            chroma_db_path = args[i + 1]
            i += 2
        elif args[i] == "--ollama-url" and i + 1 < len(args):
            ollama_url = args[i + 1]
            i += 2
        else:
            i += 1

    _embedding = ImageEmbeddingManager(
        ollama_base_url=ollama_url,
        chroma_db_name=chroma_db_name,
        chroma_db_path=chroma_db_path,
    )

    mcp_app.run()
