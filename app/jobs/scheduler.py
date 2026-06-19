from __future__ import annotations


def run_once(job) -> object:
    return job()


def run_batch(urls: list[str], updater) -> list[object]:
    return [updater.update(url) for url in urls]
