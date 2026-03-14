# Hosting Handoff

This project is now prepared for a permanent hosted setup using:

- GitHub for source control
- Render for the web service and managed PostgreSQL

## 1. Set your local git identity

Git is not configured on this Mac yet. Set it once:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 2. Create the first commit

The repository is already initialized locally. From the project root:

```bash
git add .
git commit -m "Initial Stockport model import"
```

## 3. Create a GitHub repo and push

Create an empty GitHub repository in the browser, then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/stockport-model.git
git push -u origin main
```

## 4. Deploy to Render

1. In Render, choose `New +` -> `Blueprint`.
2. Connect the GitHub repository.
3. Render will detect [render.yaml](/Users/benmills/Stockport_Model/render.yaml) and create:
   - a web service
   - a managed PostgreSQL database

## 5. Move the local database into hosted Postgres

Export your current local database:

```bash
./scripts/export_local_database.sh
```

Then restore that dump into the hosted Postgres database using the external database URL shown in Render:

```bash
./scripts/restore_database_dump.sh artifacts/YOUR_DUMP_FILE.dump "YOUR_HOSTED_DATABASE_URL"
```

## 6. Open the permanent URL

Once the restore is complete and the Render service is healthy, use the Render service URL as the permanent hosted beta URL.

## Notes

- The hosted app is configured to run read-only by default.
- Startup runs `alembic upgrade head` automatically.
- Hosted Postgres URLs like `postgres://...` and `postgresql://...` are normalized automatically in [config/settings.py](/Users/benmills/Stockport_Model/config/settings.py).
