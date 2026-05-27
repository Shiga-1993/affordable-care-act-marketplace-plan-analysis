from run_marketplace_plan_analysis import parse_money, parse_first_copay_amount


def test_parse_money_handles_currency_and_text():
    assert parse_money("$8,700") == 8700
    assert parse_money("$0 per person") == 0
    assert parse_money("Not Applicable") is None
    assert parse_money("") is None


def test_parse_first_copay_amount_extracts_first_currency_value():
    assert parse_first_copay_amount("$25 Copay after deductible") == 25
    assert parse_first_copay_amount("No Charge") == 0
    assert parse_first_copay_amount("Not Applicable") is None
