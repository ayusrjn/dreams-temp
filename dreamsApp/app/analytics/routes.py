# dreamsApp/app/analytics/routes.py

"""
API routes for graph-based narrative structure analysis.

Endpoint:
    GET /api/analytics/graph-metrics/<user_id>
        Returns quantitative structural metrics for the user's emotional
        narrative graph (centrality, transitions, cycles, etc.).
"""

import logging
from datetime import datetime, timedelta
import re

from flask import jsonify, current_app
from flask_login import login_required
from . import bp

from dreamsApp.app.builder import build_emotion_timeline
from dreamsApp.analytics.episode_segmentation import segment_timeline_to_episodes
from dreamsApp.analytics.temporal_narrative_graph import build_narrative_graph
from dreamsApp.analytics.graph_analysis import analyze_narrative_graph


logger = logging.getLogger(__name__)

# Security: only allow reasonable user_id values (letters, digits, _.-@)
_USER_ID_RE = re.compile(r'^[\w.\-@]{1,64}$')

# Default thresholds — may be made configurable via app.config in the future.
# gap_threshold:       posts further apart than this form separate episodes.
# adjacency_threshold: episodes closer than this get connected by an edge.
#                      Set to 7 days to handle typical real-world posting
#                      cadences (daily or every few days).
DEFAULT_GAP_THRESHOLD = timedelta(hours=24)
DEFAULT_ADJACENCY_THRESHOLD = timedelta(days=7)


@bp.route('/graph-metrics/<string:user_id>', methods=['GET'])
@login_required
def graph_metrics(user_id: str):
    """
    Compute and return structural graph metrics for a user's emotional narrative.

    Response 200:
        JSON with ``user_id`` and ``metrics`` (graph_summary, node_metrics,
        pattern_analysis).
    Response 404:
        No posts found for the given user_id.
    Response 500:
        Internal error during analysis (details logged server-side only).
    """
    try:
        if not _USER_ID_RE.match(user_id):
            logger.warning(f"Invalid user_id format: {user_id}")
            return jsonify({'error': 'Invalid user_id format'}), 400

        mongo = current_app.mongo['posts']

        # Fetch posts sorted chronologically
        user_posts = list(
            mongo.find({'user_id': user_id}).sort('timestamp', 1)
        )

        if not user_posts:
            return jsonify({
                'error': 'No posts found for user',
                'user_id': user_id,
            }), 404

        # Transform MongoDB documents into builder-compatible records
        records = []
        for post in user_posts:
            ts = post['timestamp']
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts))

            sentiment = post.get('sentiment', {})
            records.append({
                'timestamp': ts,
                'emotion_label': sentiment.get('label', 'neutral'),
                'score': sentiment.get('score'),
                'source_id': str(post.get('_id', '')),
            })

        # Build the analysis pipeline
        timeline = build_emotion_timeline(
            subject_id=user_id,
            records=records,
        )
        episodes = segment_timeline_to_episodes(
            timeline,
            gap_threshold=DEFAULT_GAP_THRESHOLD,
        )
        narrative_graph = build_narrative_graph(
            episodes,
            adjacency_threshold=DEFAULT_ADJACENCY_THRESHOLD,
        )
        metrics = analyze_narrative_graph(narrative_graph)

        return jsonify({
            'user_id': user_id,
            'metrics': metrics,
        })

    except Exception:
        # Log the full traceback server-side; never expose it to the client.
        logger.exception(
            "Failed to compute graph metrics for user_id=%s", user_id
        )
        return jsonify({'error': 'Failed to compute graph metrics'}), 500
