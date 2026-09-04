# Poll-trigger Cloudflare Worker

Fires a `repository_dispatch` event at `jalilcdl/mlb-model` every 5 minutes,
so `.github/workflows/poll-live-signals.yml` runs on a reliable cadence
instead of GitHub Actions' own unreliable `schedule:` trigger (confirmed to
land 147-246 minutes apart in practice, not every 5).

## Deploy (one-time, via the Cloudflare dashboard -- no CLI needed)

1. Go to https://dash.cloudflare.com -> sign up free if you don't already
   have an account (no card needed for Workers' free tier).
2. Workers & Pages -> Create -> Create Worker. Give it any name (e.g.
   `mlb-poll-trigger`) -> Deploy (the default "Hello World" is fine for now).
3. Edit code -> replace the entire contents with `poll-trigger.js` from this
   folder -> Save and Deploy.
4. Settings -> Variables and Secrets -> Add:
   - Name: `GH_PAT`
   - Value: the existing fine-grained GitHub PAT (Contents/Secrets/Workflows/
     Actions permissions, already used elsewhere for this project)
   - Type: **Secret** (encrypted), not plaintext Text
5. Settings -> Trigger Events -> Cron Triggers -> Add Cron Trigger ->
   `*/5 * * * *` -> Save.

That's it -- no code to write beyond what's already in `poll-trigger.js`,
and the Worker's own free tier is generously within limits for one fetch
call every 5 minutes.

## Verify it's actually working

Cloudflare dashboard -> your Worker -> Logs (or "Begin log stream") shows
each invocation and the `repository_dispatch POST -> 204` line. On the
GitHub side, `jalilcdl/mlb-model` -> Actions -> "Poll live signals" should
show a new run with **event: repository_dispatch** roughly every 5 minutes.

## If it ever needs to change

- Rotating `GH_PAT`: update the Cloudflare secret; nothing else changes.
- The event name (`poll`) must match `.github/workflows/poll-live-signals.yml`'s
  `repository_dispatch.types` list -- if that ever changes, update both places.
