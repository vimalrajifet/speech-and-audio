"""
Project 69: Meeting & Speech Intelligence Summarizer Engine
Extracts executive summaries, action items with assignees & deadlines,
key takeaways, and topic keywords from spoken transcripts.
"""

import re
from typing import List, Dict, Any


ACTION_PATTERNS = [
    r"(?P<assignee>[A-Z][a-z]+|team|students)\s+(?:must|should|needs to|will|is required to)\s+(?P<task>[^.,;]+?)(?:\s+by\s+|\s+before\s+|\s+on\s+)(?P<deadline>[^.,;]+)",
    r"(?:need|needs)\s+(?P<assignee>[A-Z][a-z]+|team)\s+to\s+(?P<task>[^.,;]+?)(?:\s+by\s+|\s+before\s+|\s+on\s+)(?P<deadline>[^.,;]+)",
    r"(?:please|kindly)\s+(?P<task>[^.,;]+?)(?:\s+by\s+|\s+before\s+|\s+on\s+)(?P<deadline>[^.,;]+)",
    r"(?P<assignee>[A-Z][a-z]+)\s+(?:will|shall)\s+(?P<task>[^.,;]+)"
]

HIGH_PRIORITY_KEYWORDS = ["immediately", "asap", "critical", "security", "friday", "urgent", "divergence"]


def extract_action_items(transcript: str) -> List[Dict[str, Any]]:
    """Identifies actionable tasks, designated owners, and delivery timelines."""
    actions = []
    sentences = re.split(r'(?<=[.?!])\s+', transcript)

    for sent in sentences:
        clean_sent = sent.strip()
        matched = False
        for pattern in ACTION_PATTERNS:
            match = re.search(pattern, clean_sent, re.IGNORECASE)
            if match:
                groups = match.groupdict()
                assignee = groups.get("assignee", "Unassigned").strip()
                task = groups.get("task", clean_sent).strip()
                deadline = groups.get("deadline", "TBD").strip()

                # Determine priority
                priority = "Normal"
                if any(kw in clean_sent.lower() for kw in HIGH_PRIORITY_KEYWORDS):
                    priority = "High"

                actions.append({
                    "task": task[0].upper() + task[1:] if task else "Action item",
                    "assignee": assignee.capitalize() if assignee else "Team",
                    "deadline": deadline.capitalize() if deadline else "Upcoming sprint",
                    "priority": priority,
                    "status": "Pending",
                    "source_sentence": clean_sent
                })
                matched = True
                break

        # Fallback keyword match if explicit pattern didn't match
        if not matched and any(w in clean_sent.lower() for w in ["must", "deliver", "homework", "action item"]):
            actions.append({
                "task": clean_sent,
                "assignee": "Team",
                "deadline": "End of week",
                "priority": "Medium",
                "status": "Pending",
                "source_sentence": clean_sent
            })

    return actions


def generate_executive_summary(transcript: str) -> str:
    """Produces a clean executive summary of the speech transcript."""
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', transcript) if len(s.strip()) > 10]
    if len(sentences) <= 2:
        return transcript

    # High-signal sentence extraction
    lead = sentences[0]
    core_points = [
        s for s in sentences[1:-1]
        if any(term in s.lower() for term in ["objective", "goal", "explored", "deliverable", "remember", "overall", "first", "second"])
    ]
    if not core_points:
        core_points = sentences[1:min(3, len(sentences))]

    closing = sentences[-1] if len(sentences) > 2 else ""

    summary = f"{lead} {' '.join(core_points[:2])} {closing}".strip()
    return summary


def extract_key_takeaways(transcript: str) -> List[str]:
    """Derives high-level takeaway bullets from transcript."""
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', transcript) if len(s.strip()) > 15]
    takeaways = []
    for s in sentences:
        s_lower = s.lower()
        if any(k in s_lower for k in ["objective", "explored", "gradient", "latency", "deliverable", "remember", "overall", "platform"]):
            takeaways.append(s.rstrip("."))
    if not takeaways and sentences:
        takeaways = [s.rstrip(".") for s in sentences[:3]]
    return takeaways


def extract_keywords_and_topics(transcript: str) -> List[Dict[str, Any]]:
    """Extracts topic tags and keyword frequency weights."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "we", "our",
        "all", "today", "team", "good", "morning", "let", "us", "this", "that"
    }
    words = re.findall(r'[A-Za-z]{4,}', transcript.lower())
    counts = {}
    for w in words:
        if w not in stop_words:
            counts[w] = counts.get(w, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
    return [{"topic": word.capitalize(), "relevance": round(count / max(1, len(words)), 3), "count": count} for word, count in sorted_words]


def analyze_speech_content(transcript: str, duration_sec: float = 0.0) -> Dict[str, Any]:
    """Generates complete meeting intelligence report from transcript."""
    words = transcript.split()
    word_count = len(words)
    wpm = round((word_count / (duration_sec / 60)), 1) if duration_sec > 0 else 145.0

    exec_summary = generate_executive_summary(transcript)
    summary_words = len(exec_summary.split())
    compression_ratio = round((1.0 - (summary_words / max(1, word_count))) * 100, 1)

    takeaways = extract_key_takeaways(transcript)
    action_items = extract_action_items(transcript)
    topics = extract_keywords_and_topics(transcript)

    return {
        "transcript": transcript,
        "executive_summary": exec_summary,
        "takeaways": takeaways,
        "action_items": action_items,
        "topics": topics,
        "metrics": {
            "total_words": word_count,
            "summary_words": summary_words,
            "compression_ratio": f"{max(0.0, compression_ratio)}%",
            "speaking_rate_wpm": wpm,
            "estimated_reading_time": f"{max(1, round(word_count / 200))} min"
        }
    }
