{% extends 'reveal' %}

{% block header %}
  {{ super() }}
  <style>
    .reveal .slides {
      width: 100% !important;
      font-size: 0.8em !important;
    }

    .reveal section table {
      display: block;
      overflow-x: auto;
      white-space: nowrap;
    }

    .reveal section td, 
    .reveal section th {
      font-size: 0.9em !important;
      padding: 0.4em 0.6em;
    }
  </style>
{% endblock header %}

