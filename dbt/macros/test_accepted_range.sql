{# Generic test: column values within [min_value, max_value] (nulls allowed unless also tested not_null). #}
{% test accepted_range(model, column_name, min_value=none, max_value=none, where=none) %}
select * from {{ model }}
where {{ column_name }} is not null
  {% if where %} and {{ where }} {% endif %}
  and ( false
    {% if min_value is not none %} or {{ column_name }} < {{ min_value }} {% endif %}
    {% if max_value is not none %} or {{ column_name }} > {{ max_value }} {% endif %}
  )
{% endtest %}
