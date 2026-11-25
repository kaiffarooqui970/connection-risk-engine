"""Phase 1 EDA: reproducible stats + figures for notebooks/01_eda.md. Run: uv run python notebooks/01_eda.py"""

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cre.config import eva_to_station, path  # noqa: E402

FIG = path("reports/figures/x").parent
c = duckdb.connect()
evas = ",".join(f"'{e}'" for e in eva_to_station())
c.execute(rf"""
create view r as select *,
  regexp_extract(id, '^(-?\d+)-(\d{{10}})-(\d+)$', 1) || '-' || regexp_extract(id, '^(-?\d+)-(\d{{10}})-(\d+)$', 2) run_id,
  date_diff('minute', arrival_planned_time, arrival_change_time) arr_delay,
  date_diff('minute', departure_planned_time, departure_change_time) dep_delay,
  eva in ({evas}) is_hub
from 'data/raw/stops_*.parquet'""")


def q(sql):
    return c.execute(sql).fetchall()


def show(title, sql):
    print(f"\n## {title}")
    for row in q(sql):
        print(row)


show("daily row counts at hubs: min/median/max and days below 50% of median", """
 with d as (select cast(coalesce(arrival_planned_time, departure_planned_time) as date) d, count(*) n
            from r where is_hub group by 1)
 select min(n), median(n), max(n), count(*) filter (where n < 0.5*(select median(n) from d)), count(*) from d""")
show("lowest-volume hub days", """select cast(coalesce(arrival_planned_time, departure_planned_time) as date) d, count(*) n
 from r where is_hub group by 1 order by 2 limit 6""")
show("arrival delay at hubs (non-cancelled): quantiles 0.5/0.9/0.95/0.99 , share>=5, share>=15, share<0",
     """select approx_quantile(arr_delay,[0.5,0.9,0.95,0.99]), avg((arr_delay>=5)::int), avg((arr_delay>=15)::int), avg((arr_delay<0)::int)
     from r where is_hub and not arrival_is_canceled and arr_delay is not null""")
show("extreme delays (|d|>300 or <-30) count", "select count(*) filter (where arr_delay>300), count(*) filter (where arr_delay<-30) from r")
show("cancellation share at hubs (arr, dep)", "select avg(arrival_is_canceled::int), avg(departure_is_canceled::int) from r where is_hub")
show("cancelled arrivals that still carry a change time", "select avg((arrival_change_time is not null)::int) from r where arrival_is_canceled")
show("DST fall-back night 2026-10-25 01:30-03:30 rows", "select count(*) from r where arrival_planned_time between '2026-10-25 01:30' and '2026-10-25 03:30'")
show("dep before arr at same stop (actual)", "select count(*) from r where departure_change_time < arrival_change_time")
show("same physical train at hub twice with different run ids (train_number, planned dep)", """
 select count(*) from (select train_number, departure_planned_time, eva from r where is_hub and departure_planned_time is not null
 group by all having count(distinct run_id)>1)""")

# --- figures ---
d = [x[0] for x in q("select arr_delay from r where is_hub and not arrival_is_canceled and arr_delay between -5 and 180")]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(d, bins=range(-5, 61), color="#3b6ea5")
ax[0].set(title="Arrival delay at hubs (min), linear", xlabel="minutes")
ax[1].hist(d, bins=range(-5, 181, 2), color="#3b6ea5", log=True)
ax[1].set(title="Same, log y: heavy right tail", xlabel="minutes")
fig.tight_layout()
fig.savefig(FIG / "eda_delay_distribution.png", dpi=130)

cat = """case when train_type in ('ICE','IC','EC','EN','NJ','FLX','RJ','RJX','D','ECE') then 'long-distance'
          when train_type = 'S' then 'S-Bahn' when train_type in ('Bus','SEV') then 'bus' else 'regional' end"""
rows = q(f"""select hour(arrival_planned_time) h, {cat} k, avg(arr_delay), avg((arr_delay>=5)::int) from r
 where is_hub and not arrival_is_canceled and arr_delay is not null group by all order by 1""")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for k in ["long-distance", "regional", "S-Bahn"]:
    xs = [x for x in rows if x[1] == k]
    ax[0].plot([x[0] for x in xs], [x[3] for x in xs], marker=".", label=k)
ax[0].set(title="Share arriving ≥5 min late, by hour", xlabel="planned hour")
ax[0].legend()
rows = q("""select isodow(arrival_planned_time) w, avg((arr_delay>=5)::int) from r
 where is_hub and not arrival_is_canceled and arr_delay is not null group by 1 order by 1""")
ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], [x[1] for x in rows], color="#3b6ea5")
ax[1].set(title="Share ≥5 min late, by weekday")
rows = q("""select train_type, avg((arr_delay>=5)::int) p, count(*) n from r where is_hub and not arrival_is_canceled and arr_delay is not null
 group by 1 having n > 2000 order by p desc""")
ax[2].barh([x[0] for x in rows], [x[1] for x in rows], color="#3b6ea5")
ax[2].invert_yaxis()
ax[2].set(title="Share ≥5 min late, by train type (n>2000)")
fig.tight_layout()
fig.savefig(FIG / "eda_delay_by_time_type.png", dpi=130)

rows = q(f"""with x as (select run_id, train_line_station_num k, {cat} cat, coalesce(arr_delay, dep_delay) dl,
   (train_line_station_num - 1.0) / nullif(max(train_line_station_num) over (partition by run_id) - 1, 0) pos
   from r where not arrival_is_canceled and not departure_is_canceled)
 select cat, round(pos*10)/10 b, avg(dl), approx_quantile(dl, 0.9) from x where pos is not null and dl between -30 and 600
 group by all order by 1,2""")
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, col in zip(["long-distance", "regional", "S-Bahn"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    xs = [x for x in rows if x[0] == k]
    ax.plot([x[1] for x in xs], [x[2] for x in xs], marker="o", color=col, label=f"{k} mean")
    ax.plot([x[1] for x in xs], [x[3] for x in xs], ls="--", alpha=.6, color=col, label=f"{k} p90")
ax.set(title="Delay along the journey", xlabel="relative position in run (0=origin, 1=terminus)", ylabel="minutes")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "eda_delay_along_journey.png", dpi=130)
print("\nfigures written to", FIG)
