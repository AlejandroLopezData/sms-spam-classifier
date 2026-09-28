from spam_detection.features import TextStats, clean_text


def test_clean_text_replaces_entities_with_tokens():
    out = clean_text("WIN £1000 now!!! Visit http://x.com or mail a@b.com call 07123456789")
    for token in ("urltoken", "emailtoken", "phonetoken", "currencytoken"):
        assert token in out
    assert "http" not in out
    assert "!" not in out


def test_text_stats_values_and_shape():
    stats = TextStats(log_transform=False)
    X = stats.fit_transform(["FREE entry!!! Call 07123456789 £5", "ok see you"])
    assert X.shape == (2, len(TextStats.FEATURE_NAMES))

    row = dict(zip(stats.get_feature_names_out(), X[0]))
    assert row["num_exclamations"] == 3
    assert row["has_phone"] == 1
    assert row["num_symbols"] == 1  # the £ sign
    assert dict(zip(stats.get_feature_names_out(), X[1]))["has_phone"] == 0