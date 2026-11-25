-- Hourly weather per hub. ts_utc = end of the observation hour (Bright Sky convention).
select
    station_key,
    ts                              as ts_utc,
    temperature                     as temperature_c,
    precipitation                   as precipitation_mm,
    wind_speed                      as wind_speed_kmh,
    wind_gust_speed                 as wind_gust_kmh,
    condition
from {{ source('raw', 'weather') }}
