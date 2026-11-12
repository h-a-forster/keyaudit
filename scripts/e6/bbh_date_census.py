"""date_understanding: one 'today' per stem, read by hand (table below), then every question computed."""
import json, re, sys, calendar
from datetime import date, timedelta
d = json.load(open(sys.argv[1], encoding="utf-8"))["examples"]
T = {"Today is Christmas Eve of 1937.":date(1937,12,24),"In the UK, people usually put the day before the month when formatting the date. Therefore, today is 02/01/1987 to them.":date(1987,1,2),
"Jane and John married on Jan 2, 1958. It is their 5-year anniversary today.":date(1963,1,2),"Jane scheduled 3 apointments with 5 poeple for tomorrow (Tue, 7/9/1972).":date(1972,7,8),
"Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924.":date(1924,10,15),"Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today.":date(2010,2,16),
"On May 9th, 2017 Jane bought 40 eggs. She ate one per day. Today she ran out of eggs.":date(2017,6,18),"Tomorrow is 11/12/2019.":date(2019,11,11),
"Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. Jane is correct.":date(2019,6,18),"The first day of 2019 is a Tuesday, and today is the first Monday of 2019.":date(2019,1,7),
"Yesterday was 12/31/1929. Today could not be 12/32/1929 because December has only 31 days.":date(1930,1,1),"Today is Apr 10, 1985. Jane's appointment will be 3 days later.":date(1985,4,10),
"Jane quited her job on Mar 20, 2020. 176 days have passed since then.":date(2020,9,12),"Today is the palindrome day of 2020, because the MMDDYYYY format of the date is the same backwards as forwards.":date(2020,2,2),
"May 6, 1992 is like yesterday to Jane, but that is actually ten years ago.":date(2002,5,6),"Today is Sep 9, 1909.":date(1909,9,9),"The current local time is 3:02 pm of 5/4/2004.":date(2004,5,4),
"Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings.":date(2011,1,22),"Jane is celebrating the last day of Jan 2012.":date(2012,1,31),"2015 is coming in 36 hours.":date(2014,12,30),
"Jane booked a flight for tomorrow, Jul 29, 2002.":date(2002,7,28),"The deadline is Jun 1, 2021, which is 2 days away from now.":date(2021,5,30),"Yesterday was April 30, 2021.":date(2021,5,1),
"Today is the second day of the third month of 1966.":date(1966,3,2),"This is the last day of 1899.":date(1899,12,31),"It is 4/19/1969 today.":date(1969,4,19),
"Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday.":date(2017,2,28),"Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. John is correct.":date(2019,6,19),
"Today, 8/3/1997, is a day that we will never forget.":date(1997,8,3),"The concert was scheduled to be on 06/01/1943, but was delayed by one day to today.":date(1943,6,2),
"Jane and John married on Jan 2, 1958. Today is their golden wedding anniversary.":date(2008,1,2),"Today is 9/7. Jane is watching NFL 2003.":date(2003,9,7),
"Jane got her job in 2016. Today is her 3-year work anniversary. She still remember that on Dec 2, her second day at work, she spilled coffee on her laptop.":date(2019,12,1),
"Jane was born on the last day of Feburary in 2000. Today is her 16-year-old birthday.":date(2016,2,29),"Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later.":date(2002,3,12),
"The day before yesterday was 11/23/1933.":date(1933,11,25),"Today is 3/5, and it is Jane's second time in the year 1973 to see a meteor shower.":date(1973,3,5),
"It was Sept. 1st, 2021 a week ago.":date(2021,9,8),"In the US, Thanksgiving is on the fourth Thursday of November. Today is the US Thanksgiving of 2001.":date(2001,11,22),
"Today is the last day of the first quarter of 2008.":date(2008,3,31),"Today is the first day of 2007.":date(2007,1,1)}
def shift_month(t, n):
    y, m = t.year, t.month - n
    while m < 1: m += 12; y -= 1
    return date(y, m, min(t.day, calendar.monthrange(y, m)[1]))
def ans(t, q):
    if q.startswith("today"): return t
    if q.startswith("tomorrow") or q.startswith("24 hours later"): return t + timedelta(1)
    if q.startswith("yesterday"): return t - timedelta(1)
    if q.startswith("10 days ago"): return t - timedelta(10)
    if q.startswith("one week ago"): return t - timedelta(7)
    if q.startswith("one week from"): return t + timedelta(7)
    if q.startswith("a month ago"): return shift_month(t, 1)
    if q.startswith("one year ago"): return shift_month(t, 12)
    raise ValueError(q)
bad = []; ok = 0
for i, e in enumerate(d, 1):
    a, b = e["input"].split("What is the date")
    stem = a.strip(); q = b.strip()
    t = T[stem]; x = ans(t, q).strftime("%m/%d/%Y")
    ops = dict(re.findall(r"\(([A-Z])\) (\S+)", e["input"]))
    key = ops[e["target"].strip("()")]
    if key == x: ok += 1; continue
    has = [k for k, v in ops.items() if v == x]
    bad.append((i, stem[:55], q.split(" in ")[0], "key", key, "computed", x, "option" if has else "NO OPTION"))
print("n", len(d), "agree", ok, "disagree", len(bad))
for b in bad: print(b)
import collections
print(collections.Counter(b[1] for b in bad))
