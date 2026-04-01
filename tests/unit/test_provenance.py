from evidence.provenance import badges_for_refs


def test_badges_include_source_type_and_refs() -> None:
    badges = badges_for_refs("hard_coded", ["study:demo"])
    labels = [badge.label for badge in badges]
    assert "source:hard-coded" in labels
    assert "study:demo" in labels

