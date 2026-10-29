from f1_simulator.demo import _race_presentation


def test_full_track_overview_precedes_countdown():
    assert _race_presentation(1.0) == (None, 0, False)
    assert _race_presentation(2.0) == (None, 0, False)
    assert _race_presentation(2.99) == (None, 0, False)


def test_race_light_countdown_counts_three_two_one():
    assert _race_presentation(3.0) == ("3", 1, False)
    assert _race_presentation(4.0) == ("2", 3, False)
    assert _race_presentation(5.0) == ("1", 5, False)


def test_green_lights_go_away_and_finish_message_stays():
    assert _race_presentation(6.0) == ("GO!", 5, True)
    assert _race_presentation(6.9) == (None, 0, False)
    assert _race_presentation(20.0, race_finished=True) == (
        "FINISH - ONE LAP COMPLETE",
        5,
        True,
    )
