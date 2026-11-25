-- EDA #2 regression: a cancelled event must never carry an actual time.
select * from {{ ref('fct_stop_events') }}
where (arr_cancelled and arr_actual is not null) or (dep_cancelled and dep_actual is not null)
