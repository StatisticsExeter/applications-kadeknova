# course/intro/python_exercises.py
import datetime


def parse_date(date_string):
    return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()


def sum_list(lst):
    return sum(lst)


def max_value(lst):
    return max(lst)


def reverse_string(s):
    return s[::-1]


def filter_even(lst):
    return [x for x in lst if x % 2 == 0]


def get_fifth_row(df):
    return df.iloc[4]


def column_mean(df, column):
    return df[column].mean()


def lookup_key(d, key):
    return d.get(key)


def count_occurrences(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts


def list_to_string(lst):
    return ','.join(map(str, lst))
