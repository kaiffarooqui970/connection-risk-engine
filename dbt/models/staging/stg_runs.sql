-- One row per run: origin/terminus, size, and the service date (date of first planned event, local time).
select
    run_id,
    any_value(train_type)                                      as train_type,
    any_value(train_category)                                  as train_category,
    any_value(train_number)                                    as train_number,
    min(stop_seq)                                              as first_seq,
    max(stop_seq)                                              as last_seq,
    count(*)                                                   as n_stops,
    min(coalesce(dep_planned, arr_planned))                    as run_start_planned,
    max(coalesce(arr_planned, dep_planned))                    as run_end_planned,
    cast(min(coalesce(dep_planned, arr_planned)) as date)      as service_date
from {{ ref('stg_stops') }}
group by run_id
