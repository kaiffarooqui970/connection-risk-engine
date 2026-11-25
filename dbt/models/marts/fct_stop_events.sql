-- MART: one row per (run, stop) with planned/actual times, delays, cancellation, run context.
-- Platform is NOT available in the processed source (see notebooks/01_eda.md) and is kept as an explicit
-- null column so consumers don't silently assume it exists.
with s as (select * from {{ ref('stg_stops') }}),
r as (select * from {{ ref('stg_runs') }}),
-- EDA #9: the same physical departure under two run ids -> keep one (lowest run_id), deterministic
dedup as (
    select *, row_number() over (
        partition by eva, train_number, coalesce(dep_planned, arr_planned) order by run_id) as dup_rank
    from s
)
select
    s.stop_key, s.run_id, s.stop_seq, s.eva, s.station_key, s.station_name,
    s.train_type, s.train_category, s.train_number, s.line_number, s.final_destination_station,
    s.arr_planned, s.arr_actual, s.dep_planned, s.dep_actual,
    date_diff('minute', s.arr_planned, s.arr_actual)          as arr_delay_min,
    date_diff('minute', s.dep_planned, s.dep_actual)          as dep_delay_min,
    s.arr_cancelled, s.dep_cancelled, s.is_replacement_train,
    s.station_key <> 'other'                                   as is_hub,
    r.first_seq, r.last_seq, r.n_stops, r.service_date,
    (s.stop_seq - r.first_seq)::double / nullif(r.last_seq - r.first_seq, 0) as run_progress,
    cast(null as varchar)                                      as platform
from dedup s
join r using (run_id)
where s.dup_rank = 1
