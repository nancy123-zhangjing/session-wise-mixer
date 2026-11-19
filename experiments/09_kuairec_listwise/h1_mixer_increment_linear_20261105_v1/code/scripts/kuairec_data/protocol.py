"""Pure list construction; all time arithmetic uses exact integer microseconds."""
from collections import Counter, deque

SCALE = 1_000_000


def split_at(timestamp, cuts):
    return 0 if timestamp < cuts[0] else 1 if timestamp < cuts[1] else 2


def user_samples(events, config, cuts, user_map, item_map, suppress=False):
    """Yield samples plus final counters, without ever filtering/repacking blocks.

    `events` includes valid-time quarantined records as barriers. Equal-time order
    is only a storage order: no emitted history/target contains tied records.
    """
    stats = Counter()
    history = deque(maxlen=config['history_max'])
    sessions = []
    start = 0
    gap = config['session_gap_seconds'] * SCALE
    for i in range(1, len(events)):
        if events[i]['timestamp_us'] - events[i - 1]['timestamp_us'] > gap:
            sessions.append(events[start:i])
            start = i
    if events:
        sessions.append(events[start:])
    length = config['list_length']
    for session in sessions:
        stats['sessions'] += 1
        session_id = f"u{session[0]['user_id']}:e{min(e['source_event_id'] for e in session)}"
        first, last = session[0]['timestamp_us'], session[-1]['timestamp_us']
        crossing = any(first < c <= last for c in cuts)
        stats['crossing_sessions'] += int(crossing)
        stats['tail_events'] += len(session) % length
        for offset in range(0, len(session), length):
            block = session[offset:offset + length]
            if len(block) < length:
                reason = 'tail_blocks'
            else:
                stats['full_blocks'] += 1
                if suppress:
                    reason = 'blocked_user_blocks'
                elif crossing:
                    reason = 'crossing_session_blocks'
                elif any(not e['event_valid'] or e['ambiguous'] for e in block):
                    reason = 'quarantined_blocks'
                elif len({e['video_id'] for e in block}) != length:
                    reason = 'repeated_target_item_blocks'
                elif any(e['play_duration'] <= 0 for e in block):
                    reason = 'nonplayed_blocks'
                elif block[-1]['timestamp_us'] - block[0]['timestamp_us'] > config['max_target_span_seconds'] * SCALE:
                    reason = 'excess_span_blocks'
                elif len(history) < config['history_min']:
                    reason = 'short_history_blocks'
                else:
                    reason = 'retained_blocks'
                    split = split_at(first, cuts)
                    past = list(history)
                    assert past[-1]['timestamp_us'] < block[0]['timestamp_us']
                    row = {
                        'sample_id': f"{session_id}:b{offset // length}",
                        'user_id': user_map[block[0]['user_id']],
                        'original_user_id': block[0]['user_id'],
                        'session_id': session_id,
                        'session_start_us': first, 'session_end_us': last,
                        'block_offset': offset, 'split': ('train', 'validation', 'test')[split],
                        'target_span_us': block[-1]['timestamp_us'] - block[0]['timestamp_us'],
                        'target_total_play_ms': sum(e['play_duration'] for e in block),
                        'train_optional_ge60s': sum(e['play_duration'] for e in block) >= config['train_optional_total_play_ms'] if split == 0 else None,
                    }
                    for prefix, group in [('history', past), ('target', block)]:
                        row[prefix + '_item_ids'] = [item_map[e['video_id']] for e in group]
                        row[prefix + '_original_item_ids'] = [e['video_id'] for e in group]
                        for key in ('source_event_id', 'timestamp_us', 'play_duration', 'video_duration', 'watch_ratio'):
                            row[prefix + '_' + key] = [e[key] for e in group]
                        row[prefix + '_mask'] = [True] * len(group)
                    stats['retained_' + row['split']] += 1
                    yield row
            stats[reason] += 1
            # Bad target blocks remain observed history; never remove them first.
            history.extend(e for e in block if e['event_valid'] and not e['ambiguous'] and e['play_duration'] > 0)
    yield {'_stats': dict(stats)}
