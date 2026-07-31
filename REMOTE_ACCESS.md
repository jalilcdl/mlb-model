# Accessing the dashboard from your phone at work

This gets the dashboard onto your phone over **Tailscale** — a private, encrypted
mesh between your own devices. Your phone reaches your desktop over cell data, so
the **work network is never involved** and can't block it. Nothing is exposed to
the public internet.

## What you're signing up for (the honest version)

- **Your desktop has to stay on and awake** the whole time you want access. If it
  sleeps or shuts down, the dashboard is unreachable. See "Keep the desktop awake"
  below.
- **Data only refreshes when the dashboard runs.** The projections are as current
  as the last pull — use the sidebar "🔄 Refresh probable pitchers" button from
  your phone before trusting a pick, exactly as you would at the desk.
- One-time setup is ~10 minutes. After that it's just: leave the desktop running
  the dashboard, open a link on your phone.

## One-time setup

### 1. Install Tailscale on this desktop
- Download from https://tailscale.com/download/windows and install.
- Sign in (Google/Microsoft/email — your choice). This is **your** account; I
  can't create it for you.
- Leave it running (it sits in the system tray).

### 2. Install Tailscale on your phone
- Get the "Tailscale" app from the App Store / Play Store.
- Sign in with the **same account** you used on the desktop.
- Toggle it **on**. Your two devices can now see each other privately.

### 3. Find your desktop's Tailscale address
Open a terminal on the desktop and run:
```
tailscale ip -4
```
You'll get something like `100.101.102.103`. That's your desktop's private
address on the tailnet — write it down. (It stays the same across reboots.)

Tip: Tailscale also gives each machine a name (MagicDNS). If enabled, you can use
`http://your-pc-name:8501` instead of the numeric IP — check the machine name in
the Tailscale admin console or the app.

### 4. Start the dashboard in remote mode
On the desktop, double-click **`run_dashboard_remote.bat`** (not the normal
`run_dashboard.bat` — the remote one binds so other devices can reach it).

- The **first time**, Windows Firewall will pop up asking whether to allow
  Python to accept connections. **Allow it on Private networks.** (If you miss
  the prompt, the phone won't be able to connect — see Troubleshooting.)
- Keep that window open. Closing it stops remote access.

### 5. Open it on your phone
With Tailscale toggled on, open your phone's browser to:
```
http://<your-desktop-tailscale-ip>:8501
```
e.g. `http://100.101.102.103:8501`. Bookmark it / add to home screen.

That's it.

## Keep the desktop awake

If the PC sleeps, access dies. Either:
- **Settings → System → Power → Screen and sleep →** set "When plugged in, put my
  device to sleep after" to **Never** (at least while you're relying on it), or
- Run the batch file and just leave the machine on.

The screen can turn off — that's fine. It's *sleep/hibernate* that breaks it.

## Optional: add a password

Tailscale already means only your own signed-in devices can connect, so a password
is belt-and-suspenders here — but if you want one (or if you ever switch to a
public link), edit `run_dashboard_remote.bat` and uncomment/set:
```
set MLB_DASHBOARD_PASSWORD=your-password-here
```
Save, relaunch, and the dashboard will ask for it before loading. Remove the line
to turn the gate back off.

## Troubleshooting

- **Phone can't connect / page won't load**
  - Is Tailscale toggled **on** on the phone? (It silently turns off sometimes.)
  - Is `run_dashboard_remote.bat` still running on the desktop?
  - Firewall: re-run and watch for the "Allow Python" prompt, or manually allow
    `venv\Scripts\python.exe` through Windows Defender Firewall on Private
    networks.
- **Worked before, now it doesn't** — the desktop probably went to sleep, or its
  IP address changed (rare). Re-check `tailscale ip -4`.
- **It's slow on cell data** — the projections run 10,000 simulations per game;
  drop "Simulations per game" in the sidebar to e.g. 3,000 for a snappier phone
  experience. The numbers barely move.

## Why not just host it online?

A hosted public website (Streamlit Cloud, a tunnel, etc.) would let a locked-down
work computer reach it and wouldn't need the desktop on — but it means putting the
code on GitHub, reworking how data and odds get in, adding real auth, and it has
some feasibility unknowns (cloud IPs sometimes get blocked by the stats sources).
For "check it on my phone at work," Tailscale is simpler, private, and free. If you
later want the always-on hosted version, that's a separate build we can do.
