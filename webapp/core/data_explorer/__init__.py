"""
Data Explorer - unified query and visualization layer for stock_data.db.

Modules:
  feature_expander  - expand features_json column to flat columns
  chart_suggester   - heuristic chart-type picker based on DataFrame shape
  schema_discovery  - scan sqlite_master and classify tables
  query_runner      - SELECT-only SQL execution with LIMIT + timeout
  query_store       - CRUD for saved_queries in webapp.db
"""
