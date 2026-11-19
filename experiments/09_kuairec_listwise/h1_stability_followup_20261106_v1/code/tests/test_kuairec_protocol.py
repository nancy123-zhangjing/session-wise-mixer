"""Behavioral contract tests for five-item continuation and exact CSV parsing."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kuairec_data.prepare import FIELDS as EVENT_FIELDS, ingest, parse_chunk
from scripts.kuairec_data.protocol import SCALE, split_at, user_samples
from scripts.kuairec_data.text import FIELDS as TEXT_FIELDS


CONFIG = {
    'history_max': 50, 'history_min': 5, 'session_gap_seconds': 1800,
    'list_length': 5, 'max_target_span_seconds': 600,
    'train_optional_total_play_ms': 60000,
}


def events_at(times, items=None, play=1000):
    if items is None:
        items = list(range(len(times)))
    return [dict(source_event_id=i + 1, user_id=0, video_id=item,
                 timestamp_us=time, play_duration=play, video_duration=1000,
                 watch_ratio=play / 1000, event_valid=True, ambiguous=False)
            for i, (time, item) in enumerate(zip(times, items))]


def run_events(events, cuts=(100_000 * SCALE, 200_000 * SCALE), suppress=False):
    item_map = {item: item + 1 for item in {event['video_id'] for event in events}}
    results = list(user_samples(events, CONFIG, cuts, {0: 1}, item_map, suppress=suppress))
    return results[:-1], results[-1]['_stats']


class SessionListTests(unittest.TestCase):
    def test_gap_equal_1800_stays_and_one_microsecond_more_splits(self):
        for extra, expected_sessions in ((0, 1), (1, 2)):
            with self.subTest(extra_us=extra):
                times = [i * SCALE for i in range(5)]
                times += [(1804 + i) * SCALE + extra for i in range(5)]
                rows, stats = run_events(events_at(times))
                self.assertEqual(stats['sessions'], expected_sessions)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['target_source_event_id'], [6, 7, 8, 9, 10])

    def test_target_span_equal_600_is_allowed_and_one_microsecond_more_rejected(self):
        for extra, expected in ((0, 1), (1, 0)):
            with self.subTest(extra_us=extra):
                times = [i * SCALE for i in range(5)]
                times += [x * SCALE for x in (10, 160, 310, 460, 610)]
                times[-1] += extra
                rows, stats = run_events(events_at(times))
                self.assertEqual(len(rows), expected)
                if expected:
                    self.assertEqual(rows[0]['target_span_us'], 600 * SCALE)
                else:
                    self.assertEqual(stats['excess_span_blocks'], 1)

    def test_tail_does_not_pad_or_join_next_session(self):
        times = [i * SCALE for i in range(12)] + [(3000 + i) * SCALE for i in range(5)]
        rows, stats = run_events(events_at(times))
        self.assertEqual(stats['tail_events'], 2)
        self.assertEqual(stats['tail_blocks'], 1)
        self.assertEqual([row['target_source_event_id'] for row in rows],
                         [[6, 7, 8, 9, 10], [13, 14, 15, 16, 17]])
        self.assertEqual(rows[-1]['history_source_event_id'], list(range(1, 13)))

    def test_repeated_target_block_is_not_repacked_and_stays_in_later_history(self):
        items = [0, 1, 2, 3, 4, 20, 21, 20, 22, 23, 30, 31, 32, 33, 34]
        rows, stats = run_events(events_at([i * SCALE for i in range(15)], items))
        self.assertEqual(stats['repeated_target_item_blocks'], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['target_source_event_id'], [11, 12, 13, 14, 15])
        self.assertEqual(rows[0]['history_original_item_ids'], items[:10])
        self.assertEqual(rows[0]['block_offset'], 10)

    def test_nonplayed_event_rejects_whole_block_but_other_played_events_remain_history(self):
        events = events_at([i * SCALE for i in range(15)])
        events[7]['play_duration'] = 0
        events[7]['watch_ratio'] = 0.0
        rows, stats = run_events(events)
        self.assertEqual(stats['nonplayed_blocks'], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['target_source_event_id'], [11, 12, 13, 14, 15])
        self.assertEqual(rows[0]['history_source_event_id'], [1, 2, 3, 4, 5, 6, 7, 9, 10])

    def test_history_is_latest_50_played_records_and_keeps_repeats(self):
        items = [i % 7 for i in range(60)]
        rows, _ = run_events(events_at([i * SCALE for i in range(60)], items))
        row = rows[-1]
        self.assertEqual(row['target_source_event_id'], [56, 57, 58, 59, 60])
        self.assertEqual(row['history_source_event_id'], list(range(6, 56)))
        self.assertEqual(row['history_original_item_ids'], items[5:55])
        self.assertLess(len(set(row['history_item_ids'])), len(row['history_item_ids']))

    def test_crossing_whole_session_yields_no_targets_but_provides_future_history(self):
        times = [i * SCALE for i in range(10)] + [(3000 + i) * SCALE for i in range(5)]
        rows, stats = run_events(events_at(times), cuts=(5 * SCALE, 20 * SCALE))
        self.assertEqual(stats['crossing_sessions'], 1)
        self.assertEqual(stats['crossing_session_blocks'], 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['split'], 'test')
        self.assertEqual(rows[0]['target_source_event_id'], [11, 12, 13, 14, 15])
        self.assertEqual(rows[0]['history_source_event_id'], list(range(1, 11)))

    def test_cut_time_belongs_to_later_partition(self):
        cuts = (100 * SCALE, 200 * SCALE)
        self.assertEqual([split_at(t, cuts) for t in (cuts[0] - 1, cuts[0], cuts[1] - 1, cuts[1])], [0, 1, 1, 2])

    def test_60_second_flag_is_train_only_and_never_a_main_filter(self):
        times = [i * SCALE for i in range(10)]
        times += [(3000 + i) * SCALE for i in range(10)]
        times += [(6000 + i) * SCALE for i in range(10)]
        for play, flag in ((11999, False), (12000, True)):
            with self.subTest(play=play):
                rows, _ = run_events(events_at(times, play=play), cuts=(2000 * SCALE, 5000 * SCALE))
                self.assertEqual(len(rows), 5)
                self.assertEqual([r['train_optional_ge60s'] for r in rows if r['split'] == 'train'], [flag])
                self.assertTrue(all(r['train_optional_ge60s'] is None for r in rows if r['split'] != 'train'))

    def test_padding_zero_is_reserved_not_a_real_user_or_item(self):
        rows, _ = run_events(events_at([i * SCALE for i in range(10)]))
        row = rows[0]
        self.assertEqual(row['user_id'], 1)
        self.assertEqual(row['original_user_id'], 0)
        self.assertEqual(row['history_item_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(row['history_original_item_ids'][0], 0)
        for side in ('history', 'target'):
            self.assertNotIn(0, row[side + '_item_ids'])
            self.assertEqual(row[side + '_mask'], [True] * len(row[side + '_item_ids']))

    def test_ambiguous_events_are_barriers_not_reordered_into_targets_or_history(self):
        events = events_at([i * SCALE for i in range(15)])
        events[7]['timestamp_us'] = events[6]['timestamp_us']
        events[6]['ambiguous'] = events[7]['ambiguous'] = True
        rows, stats = run_events(events)
        self.assertEqual(stats['quarantined_blocks'], 1)
        self.assertEqual(rows[0]['target_source_event_id'], [11, 12, 13, 14, 15])
        self.assertEqual(rows[0]['history_source_event_id'], [1, 2, 3, 4, 5, 6, 9, 10])

    def test_unknown_timestamp_user_suppression_prevents_targets(self):
        rows, stats = run_events(events_at([i * SCALE for i in range(15)]), suppress=True)
        self.assertEqual(rows, [])
        self.assertEqual(stats['blocked_user_blocks'], 3)


def raw_frame(timestamp_strings, index=None):
    rows = []
    for timestamp in timestamp_strings:
        seconds, _, fraction = timestamp.partition('.')
        micros = int(seconds) * SCALE + int(fraction.ljust(6, '0'))
        local = pd.Timestamp(micros, unit='us') + pd.Timedelta(hours=8)
        rows.append(dict(user_id='0', video_id='7', play_duration='2500', video_duration='1000',
                         timestamp=timestamp, watch_ratio='2.5', time=str(local), date=local.strftime('%Y%m%d')))
    return pd.DataFrame(rows, index=index)


class ExactParseTests(unittest.TestCase):
    def test_fractional_seconds_are_exact_and_do_not_create_ties(self):
        raw = raw_frame(['1600000000.123456', '1600000000.123457', '1600000000.1', '1600000001'])
        parsed, valid_time, clock_ok = parse_chunk(raw, 0)
        self.assertEqual(parsed.timestamp_us.tolist(), [1600000000123456, 1600000000123457, 1600000000100000, 1600000001000000])
        self.assertTrue(valid_time.all() and clock_ok.all() and parsed.event_valid.all())
        self.assertTrue((parsed.watch_ratio > 1).all())

    def test_nonzero_csv_chunk_index_and_offset_preserve_row_alignment(self):
        raw = raw_frame(['1600000000.123456', '1600000001.654321'], index=[100_000, 100_001])
        raw.loc[100_001, 'user_id'] = '8'
        raw.loc[100_001, 'video_id'] = '9'
        parsed, valid_time, clock_ok = parse_chunk(raw, 100_000)
        self.assertEqual(parsed.index.tolist(), [100_000, 100_001])
        self.assertEqual(parsed.source_event_id.tolist(), [100_001, 100_002])
        self.assertEqual(parsed.user_id.tolist(), [0, 8])
        self.assertEqual(parsed.video_id.tolist(), [7, 9])
        self.assertEqual(parsed.timestamp_us.tolist(), [1600000000123456, 1600000001654321])
        self.assertTrue(parsed.notna().all().all())
        self.assertTrue(valid_time.all() and clock_ok.all() and parsed.event_valid.all())

    def test_invalid_precision_is_quarantined_instead_of_rounded(self):
        raw = raw_frame(['1600000000.123456'])
        raw.loc[0, 'timestamp'] = '1600000000.1234567'
        parsed, valid_time, _ = parse_chunk(raw, 50)
        self.assertFalse(valid_time.iloc[0])
        self.assertFalse(parsed.event_valid.iloc[0])
        self.assertEqual(parsed.source_event_id.iloc[0], 51)

    def test_invalid_feedback_and_local_clock_are_not_accepted(self):
        raw = raw_frame(['1600000000.123456'] * 4)
        raw.loc[0, 'play_duration'] = '-1'
        raw.loc[1, 'video_duration'] = '0'
        raw.loc[2, 'watch_ratio'] = 'nan'
        raw.loc[3, 'time'] = str(pd.Timestamp(raw.loc[3, 'time']) + pd.Timedelta(microseconds=1))
        parsed, valid_time, clock_ok = parse_chunk(raw, 0)
        self.assertTrue(valid_time.all())
        self.assertFalse(parsed.event_valid.any())
        self.assertFalse(clock_ok.iloc[3])

    def test_date_label_mismatch_is_retained_when_two_precise_clocks_agree(self):
        raw = raw_frame(['1600000000.123456'])
        raw.loc[0, 'date'] = '20200101'
        parsed, valid_time, clock_ok = parse_chunk(raw, 0)
        self.assertTrue(valid_time.iloc[0] and clock_ok.iloc[0] and parsed.event_valid.iloc[0])
        self.assertTrue(parsed.date_mismatch.iloc[0])
        self.assertEqual(parsed.date_raw.iloc[0], '20200101')
        self.assertEqual(parsed.timestamp_raw.iloc[0], '1600000000.123456')


class IngestDeduplicationTests(unittest.TestCase):
    """Exercise real ZIP CSV chunking and disk buckets, not a copied dedup expression."""

    def setUp(self):
        base = Path(__file__).resolve().parents[1] / 'outputs' / 'kuairec_protocol_tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='ingest-', dir=base)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def baseline(self):
        # Repeated views of the same item at distinct times must all survive.
        return raw_frame([f'{1600000000 + i}.123456' for i in range(12)]).to_dict('records')

    def run_ingest(self, rows):
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=EVENT_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        archive = self.root / 'fixture.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('KuaiRec/data/big_matrix.csv', buffer.getvalue().encode('utf-8'))
        text = self.root / 'text.csv'
        with text.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(TEXT_FIELDS)
            writer.writerow([7, 'UNKNOWN', 'fixture caption', '[]', 1, 'category', -124, 'UNKNOWN', -124, 'UNKNOWN'])
        config = {'input_archive': str(archive), 'input_text': str(text),
                  'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'text_sha256': hashlib.sha256(text.read_bytes()).hexdigest(),
                  'chunk_rows': 2, 'user_buckets': 2, 'split_quantiles': [0.8, 0.9]}
        output = self.root / 'prepared'
        output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            meta = ingest(config, output)
        events = pd.concat([pq.read_table(path).to_pandas() for path in sorted((output / 'events').glob('*.parquet'))], ignore_index=True)
        duplicates = pq.read_table(output / 'duplicates.parquet').to_pandas()
        return meta, events, duplicates

    def test_date_only_difference_is_one_event_with_both_labels_in_ledger(self):
        rows = self.baseline()
        duplicate = dict(rows[0], date='20200101')
        rows.insert(4, duplicate)  # Duplicate crosses the two-row CSV chunk boundary.
        meta, events, duplicates = self.run_ingest(rows)
        self.assertEqual(meta['counts']['raw_events'], 13)
        self.assertEqual(meta['counts']['exact_duplicates_removed'], 1)
        self.assertEqual(meta['counts']['duplicate_rows_with_different_date_label'], 1)
        self.assertEqual(meta['counts']['ambiguous_events'], 0)
        self.assertEqual(len(events), 12)
        self.assertEqual(duplicates.to_dict('records'), [
            {'removed_event_id': 5, 'canonical_event_id': 1,
             'removed_date_label': '20200101', 'canonical_date_label': rows[0]['date']}])
        canonical = events.loc[events.source_event_id == 1].iloc[0]
        self.assertEqual(canonical.timestamp_us, 1600000000123456)
        self.assertTrue(canonical.event_valid)
        self.assertFalse(canonical.ambiguous)

    def test_same_item_repeated_at_distinct_times_is_not_deduplicated(self):
        meta, events, duplicates = self.run_ingest(self.baseline())
        self.assertEqual(len(events), 12)
        self.assertEqual(meta['counts']['exact_duplicates_removed'], 0)
        self.assertEqual(meta['counts']['ambiguous_events'], 0)
        self.assertTrue(duplicates.empty)
        self.assertEqual(events.video_id.nunique(), 1)
        self.assertEqual(events.timestamp_us.nunique(), 12)
        self.assertEqual(sorted(events.source_event_id.tolist()), list(range(1, 13)))

    def test_close_but_distinct_decimal_feedback_is_not_merged_by_float_rounding(self):
        rows = self.baseline()
        rows[0]['watch_ratio'] = '0.123456789012345670'
        second = dict(rows[0], watch_ratio='0.123456789012345671')
        rows.insert(4, second)
        meta, events, duplicates = self.run_ingest(rows)
        self.assertEqual(meta['counts']['exact_duplicates_removed'], 0)
        self.assertEqual(meta['counts']['ambiguous_events'], 2)
        self.assertTrue(duplicates.empty)
        conflicting = events.loc[events.timestamp_us == 1600000000123456]
        self.assertEqual(len(conflicting), 2)
        self.assertEqual(conflicting.watch_ratio.nunique(), 1)  # Float alone would lose the distinction.
        self.assertEqual(set(conflicting.watch_ratio_raw), {'0.123456789012345670', '0.123456789012345671'})
        self.assertTrue(conflicting.ambiguous.all())

    def test_invalid_clock_rows_keep_different_clock_provenance(self):
        rows = self.baseline()
        first = dict(rows[0], time=str(pd.Timestamp(rows[0]['time']) + pd.Timedelta(microseconds=1)))
        second = dict(rows[0], time=str(pd.Timestamp(rows[0]['time']) + pd.Timedelta(microseconds=2)))
        rows.extend([first, second])
        meta, events, duplicates = self.run_ingest(rows)
        self.assertEqual(meta['counts']['exact_duplicates_removed'], 0)
        self.assertEqual(meta['counts']['clock_mismatch_rows'], 2)
        self.assertTrue(duplicates.empty)
        conflicting = events.loc[events.timestamp_us == 1600000000123456]
        self.assertEqual(len(conflicting), 3)
        self.assertEqual(conflicting.time_raw.nunique(), 3)
        self.assertEqual(int(conflicting.event_valid.sum()), 1)
        self.assertTrue(conflicting.ambiguous.all())


if __name__ == '__main__':
    unittest.main()
