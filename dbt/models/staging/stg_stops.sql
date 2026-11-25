-- One row per (run, stop). Encodes the EDA fixes once, so no downstream code re-derives them:
--  * run_id from the stop id (train_line_ride_id is a recurring line id, not a run)   [EDA #1]
--  * cancelled stops have NO actual time (source keeps stale change times)            [EDA #2]
--  * actual departure = max(dep_change, arr_change) (stale departure forecasts)       [EDA #3]
with src as (
    select
        *,
        regexp_extract(id, '^(-?\d+)-(\d{10})-(\d+)$', 1) || '-' || regexp_extract(id, '^(-?\d+)-(\d{10})-(\d+)$', 2) as run_id
    from {{ source('raw', 'stops') }}
)
select
    run_id || '-' || train_line_station_num::varchar          as stop_key,
    run_id,
    train_line_station_num                                    as stop_seq,
    eva,
    coalesce(h.station_key, 'other')                          as station_key,
    station_name,
    coalesce(train_type, 'unknown')                           as train_type,
    {{ train_category('train_type') }}                        as train_category,
    train_number,
    line_number,
    final_destination_station,
    arrival_planned_time                                      as arr_planned,
    case when not arrival_is_canceled then arrival_change_time end as arr_actual,
    departure_planned_time                                    as dep_planned,
    case when not departure_is_canceled
         then greatest(departure_change_time, coalesce(arrival_change_time, departure_change_time)) end as dep_actual,
    arrival_is_canceled                                       as arr_cancelled,
    departure_is_canceled                                     as dep_cancelled,
    coalesce(is_replacement_train, false)                     as is_replacement_train,
    source_month
from src
left join {{ ref('hub_evas') }} h using (eva)
