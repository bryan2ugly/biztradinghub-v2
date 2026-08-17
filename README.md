# BizTradingHub

A public, friends-shareable website showing your BTC signal model live,
plus a learning hub explaining candlesticks, signal tiers, regimes, and
how Kalshi 15-min markets work.

**This is a completely separate project from your `kalshi_bot` trading
bot** — nothing in your bot's folder was touched. `backend/price_feed.py`
and `backend/signal_engine.py` are plain copies of the read-only, no-
credentials-needed parts of your engine (public Coinbase/Kraken/Gemini
price data only). There is **no Kalshi API key anywhere in this
project**, no order placement, and nothing that could touch your Kalshi
account — this site only ever computes and displays the model's
probability read.

## Round history that survives redeploys (new)

On Render's free tier, every redeploy wipes the container's local
filesystem clean -- including `rounds_history.jsonl`. So without this,
your track record resets to empty every time you push a code update.

The fix: store round history in a JSON file in your own GitHub repo
instead, using GitHub's API. It survives redeploys because it isn't
stored in the container at all -- it's a real file in your repo.

**This is optional** -- skip it and everything still works exactly as
before, it just won't survive a Render redeploy.

To turn it on:

1. Go to github.com/settings/tokens?type=beta → "Generate new token".
2. Give it a name (e.g. "biztradinghub-history"), set an expiration
   (a year is fine), under "Repository access" pick "Only select
   repositories" → your repo. Under "Permissions" → "Repository
   permissions" → set **Contents** to **Read and write**. Generate,
   copy the token (starts with `github_pat_...`) — you won't see it
   again.
3. On Render, add two environment variables:
   - `GITHUB_TOKEN` = the token you just copied
   - `GITHUB_REPO` = `yourusername/yourreponame` (e.g. `bryan2ugly/Testing-stuff`)
4. Redeploy. From then on, every completed round also gets written to
   a `round_history_store.json` file at the root of your repo — check
   your repo after a round or two completes and you'll see it appear.

Treat that token like a password — anyone with it can write to your
repo. If it ever leaks, revoke it from the same GitHub settings page
and generate a new one.

## Log in / accounts (new)

The site now requires a login — nobody can see the dashboard without an
account. You manage accounts yourself at `/admin`.

1. **Set an admin password.** This is what unlocks `/admin` for you.
   - Locally: `export ADMIN_PASSWORD=pickAstrongPassword` in Terminal
     *before* running uvicorn (same Terminal window).
   - On Railway/Render: add it under Settings → Variables as
     `ADMIN_PASSWORD` = whatever you pick. Never put it in the code —
     that's why it's an environment variable instead.
2. Open `yoursite.com/admin`, enter that password.
3. Add a username + password for each friend. Share those with them
   individually (not the admin password — that one's yours only).
4. The admin page shows who's online right now and when each person
   was last seen. Click **Remove** next to anyone to kick them out
   immediately, even mid-session.

Two files get created automatically once you add your first user:
`backend/users.json` (accounts) and `backend/rounds_history.jsonl`
(round history, from before). **Delete both before uploading this
project to a public GitHub repo** — they're runtime data, not code,
and users.json holds your friends' (hashed) passwords.

Also worth knowing: on most free hosting tiers, a fresh deploy wipes
the container's filesystem, which resets `users.json` (you'd need to
re-add accounts) — this doesn't happen on a simple restart, only when
you push new code. If that becomes annoying, ask about adding a
persistent volume on your host.

## Run it locally

```
cd backend
export ADMIN_PASSWORD=pickAstrongPassword
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — the model will preload ~30 min of recent
BTC history (same trick your bot uses) so it doesn't sit on "warming
up," then start updating live. You'll be redirected to `/login` until
you add yourself an account via `/admin`.

## Share it with friends

Running it on your Mac only works while your Mac is on and awake.
For an always-on link you can send people, deploy `backend/` (with
`frontend/` alongside it) to a small always-on host:

- **Railway** or **Render**: point either at this folder, use
  `uvicorn server:app --host 0.0.0.0 --port $PORT` as the start
  command, and set `ADMIN_PASSWORD` under their environment variables
  settings. Free/cheap tiers are enough for this.
- A ~$5/mo VPS works too — same start command, run it under `tmux` or
  as a systemd service.

No Kalshi credentials or secrets of that kind are needed for this
deployment — the only secret is the `ADMIN_PASSWORD` you set yourself.

## What's next (optional, more advanced)

- **Live Kalshi price overlay**: Kalshi's own market-data endpoints
  (candlesticks, orderbook) don't require authentication, so you could
  add a real "model vs. Kalshi price" edge panel later without ever
  putting your API key on the public site.
- **Track record**: your engine already scores each tier's real
  accuracy (`track_record` in `signal_engine.py`) — a "how has this
  actually done" panel would slot right into the learning hub.
- **Custom domain**: any of the hosts above will front a domain you
  buy for free with a CNAME record.
