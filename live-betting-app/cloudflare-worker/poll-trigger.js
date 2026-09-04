/**
 * Cron-triggered pinger: fires a GitHub repository_dispatch event every 5
 * minutes so .github/workflows/poll-live-signals.yml runs on a reliable
 * cadence.
 *
 * WHY THIS EXISTS: GitHub Actions' own `schedule:` trigger is a documented
 * best-effort queue with no timing guarantee -- confirmed directly on this
 * repo landing 147-246 minutes apart instead of every 5, especially bad on
 * lower-activity repos. `repository_dispatch` (this Worker's job) is a
 * direct API-triggered event, not the slow polling queue, and was verified
 * empirically (3/3 trials, ~8 seconds from fire to run start) to avoid that
 * delay entirely. Cloudflare Cron Triggers are themselves a reliable,
 * first-class platform feature (not a best-effort queue), which is the
 * actual fix here -- moving the unreliable clock outside GitHub, not
 * changing what runs once triggered.
 *
 * This Worker does not touch odds, game state, or any betting logic --  it
 * only tells GitHub "run the workflow now." All OBSERVE-ONLY logic lives in
 * live-betting-app/core + sports/, unchanged by this.
 *
 * Secret required (Cloudflare dashboard -> Worker -> Settings -> Variables
 * and Secrets -> add as "Secret", not plaintext): GH_PAT
 *   The existing repo PAT already used for git push / secrets management.
 *   Deliberately reused rather than a narrower token -- a scope/security
 *   tradeoff made knowingly, not an oversight.
 */
export default {
  async scheduled(event, env, ctx) {
    const resp = await fetch(
      "https://api.github.com/repos/jalilcdl/mlb-model/dispatches",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GH_PAT}`,
          Accept: "application/vnd.github+json",
          "User-Agent": "live-betting-app-cron-worker", // GitHub's API rejects requests with no User-Agent
        },
        body: JSON.stringify({ event_type: "poll" }),
      }
    );
    // 204 = accepted. Anything else means the dispatch didn't fire --
    // logged so it's visible in the Worker's own real-time/tail logs.
    console.log(`repository_dispatch POST -> ${resp.status}`);
    if (resp.status !== 204) {
      console.log(await resp.text());
    }
  },
};
