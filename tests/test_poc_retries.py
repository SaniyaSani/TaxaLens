"""Retry transient GET failures without skipping pages or retrying refusals."""
from unittest.mock import Mock

import pytest
import requests

from scripts import fetch_poc_metadata as fetch


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(fetch.time, 'sleep', Mock())
    instance = fetch.Client(pause=0)
    instance.session.get = Mock()
    return instance


def ok():
    result = Mock(status_code=200)
    result.json.return_value = {'results': [{'key': 42}]}
    return result


def test_disconnected_page_retried_at_same_offset(client):
    client.session.get.side_effect = [requests.ConnectionError('RemoteDisconnected'), requests.Timeout(), ok()]
    params = {'offset': 12000, 'limit': 200}
    assert client.get('/occurrence/search', params) == {'results': [{'key': 42}]}
    assert client.calls == 3
    assert all(call.kwargs['params'] == params for call in client.session.get.call_args_list)
    assert [call.args[0] for call in fetch.time.sleep.call_args_list] == [0, 5, 0, 10, 0]


def test_retries_are_bounded(client):
    client.session.get.side_effect = requests.ConnectionError('disconnected')
    with pytest.raises(RuntimeError, match='after 4 attempts'):
        client.get('/occurrence/search', {'offset': 12000})
    assert client.calls == client.session.get.call_count == 4


@pytest.mark.parametrize('status', [401, 403, 429])
def test_refusals_do_not_retry(client, status):
    client.session.get.return_value = Mock(status_code=status, headers={'Retry-After': '120'})
    with pytest.raises(RuntimeError):
        client.get('/occurrence/search')
    assert client.session.get.call_count == 1


def test_ssl_error_does_not_retry(client):
    client.session.get.side_effect = requests.exceptions.SSLError('certificate failure')
    with pytest.raises(requests.exceptions.SSLError):
        client.get('/occurrence/search')
    assert client.session.get.call_count == 1


def test_retries_count_toward_request_budget(client):
    client.maximum = 1
    client.session.get.side_effect = requests.ConnectionError('disconnected')
    with pytest.raises(RuntimeError, match='Request budget reached'):
        client.get('/occurrence/search')
    assert client.session.get.call_count == 1
