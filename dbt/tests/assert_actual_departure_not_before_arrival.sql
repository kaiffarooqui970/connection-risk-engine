-- EDA #3 regression: after the max(dep, arr) fix no stop may depart before it arrives.
select * from {{ ref('fct_stop_events') }}
where dep_actual < arr_actual
