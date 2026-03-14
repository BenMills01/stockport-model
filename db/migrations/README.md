# Migrations

Alembic is configured at the project root (`alembic/`, `alembic.ini`).

## Common commands

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back one revision
alembic downgrade -1

# Show current revision
alembic current

# Generate a new migration after editing db/schema.py
alembic revision --autogenerate -m "describe_your_change"
```

## Bootstrap (fresh database)

For a brand-new database, run `db/init_db.py` to create all tables via
`Base.metadata.create_all()`, then stamp Alembic at head so it knows
the schema is already current:

```bash
python db/init_db.py
alembic stamp head
```

Subsequent schema changes must go through `alembic revision --autogenerate`.
