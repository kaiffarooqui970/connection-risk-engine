{# Coarse train category. Operator brand codes (ABR, EB, MRB, OE, STB, ...) are regional services. #}
{% macro train_category(col) -%}
  case
    when {{ col }} in ('ICE','IC','EC','ECE','EN','NJ','FLX','RJ','RJX','D','TGV','EST') then 'long_distance'
    when {{ col }} = 'S' then 'sbahn'
    when {{ col }} in ('Bus','SEV','BSV') then 'bus'
    when {{ col }} is null then 'unknown'
    else 'regional'
  end
{%- endmacro %}
