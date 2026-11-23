"""前後ページ取得の境界・重複・出典を確認する。APIは使わない。"""
from copy import deepcopy
from contextlib import contextmanager

import pytest

from app import retrieval


def page(doc, number):
    return dict(document_id=doc, page=number, page_id=f"{doc}-p{number:03}", content=f"{doc}:{number}")


def test_neighbors_preserve_rank_and_include_full_text():
    pages = [page('A', n) for n in range(1, 7)]
    hits = [{**pages[3], 'distance': 0.1}, {**pages[1], 'distance': 0.2}]
    original = deepcopy(hits)
    result = retrieval.expand_neighbors(hits, pages)
    assert [p['page'] for p in result] == [4, 2, 3, 5, 1]
    assert result[2]['content'] == 'A:3'
    assert result[2]['neighbor_of'] == ['A-p004', 'A-p002']
    assert all('distance' not in p for p in result[2:])
    assert hits == original


@pytest.mark.parametrize('number,expected', [(1, [1, 2]), (3, [3, 2])])
def test_document_boundaries_do_not_cross(number, expected):
    pages = [page(d, n) for d in ['A', 'B'] for n in range(1, 4)]
    result = retrieval.expand_neighbors([page('A', number)], pages)
    assert [p['page'] for p in result] == expected
    assert {p['document_id'] for p in result} == {'A'}


def test_duplicate_and_adjacent_hits_are_returned_once():
    pages = [page('A', n) for n in range(1, 5)]
    result = retrieval.expand_neighbors([pages[1], pages[2], pages[1]], pages)
    assert len(result) == 4
    assert [p['retrieval_role'] for p in result[:2]] == ['seed', 'seed']
    assert len({p['page_id'] for p in result}) == 4


def test_missing_page_is_not_skipped():
    pages = [page('A', 1), page('A', 3)]
    assert retrieval.expand_neighbors([pages[0]], pages) == [{**pages[0], 'retrieval_role': 'seed'}]


def test_no_hits_and_unknown_document_do_not_expand():
    assert retrieval.expand_neighbors([], [page('A', 1)]) == []
    assert retrieval.expand_neighbors([page('B', 1)], [page('A', 1)]) == []


def test_search_option_switches_only_expansion(monkeypatch):
    pages = [page('A', n) for n in range(1, 4)]
    class FakeDB:
        def execute(self, sql, args):
            return [{**pages[1], 'distance': 0.1}]
    @contextmanager
    def fake_index(path):
        yield FakeDB()
    monkeypatch.setattr(retrieval, 'open_index', fake_index)
    monkeypatch.setattr(retrieval, 'embed_text', lambda *args: [1.0])
    monkeypatch.setattr(retrieval, 'load_pages', lambda *args: pages)
    assert len(retrieval.search('x', include_neighbors=False)) == 1
    assert len(retrieval.search('x', include_neighbors=True)) == 3
    monkeypatch.setattr(retrieval, 'RAG_INCLUDE_NEIGHBORS', True)
    assert len(retrieval.search('x')) == 3
