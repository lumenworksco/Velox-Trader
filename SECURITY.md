# Security Policy

## Secrets Management

- **Never commit API keys, secrets, or `.env` files** to the repository.
- Use environment variables for all sensitive configuration:
  - `ALPACA_API_KEY` and `ALPACA_API_SECRET`
  - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
  - `FRED_API_KEY`
  - `DASHBOARD_SECRET_KEY`
  - `POSTGRES_PASSWORD` and `GRAFANA_ADMIN_PASSWORD`
- The `.gitignore` file excludes `.env`, `bot.db`, `state.json`, and database files.
- If using Docker, pass secrets via environment variables or Docker secrets -- do not bake them into images.
- The `.env` file is loaded automatically by `config/settings.py` via `python-dotenv`.

## Reporting Security Issues

If you discover a security vulnerability, **do not open a public issue**.

Instead, please report it privately via email or GitHub Security Advisories.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Best Practices

- Run in paper trading mode (`ALPACA_LIVE=false`) until you have thoroughly tested your configuration.
- Use a dedicated API key with minimal permissions for the bot.
- Rotate API keys periodically (Alpaca dashboard, Telegram BotFather, FRED).
- Monitor the bot's activity via the dashboard (http://localhost:8080) and Grafana (http://localhost:3000).
- Set up Telegram alerts for unexpected behavior.
- Keep dependencies up to date: `pip install -r requirements.txt --upgrade`.
- Review `docs/DISCLAIMER.md` for risk disclosures before deploying with real capital.
