from app.schemas.extraction import TextBlock


def order_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    if not blocks or any(b.bbox is None for b in blocks):
        return blocks
    by_left = sorted(blocks, key=lambda b: b.bbox[0])
    right = by_left[0].bbox[2]
    for index, block in enumerate(by_left[1:], 1):
        if block.bbox[0] - right > 0.04:
            return order_blocks(by_left[:index]) + order_blocks(by_left[index:])
        right = max(right, block.bbox[2])
    return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
