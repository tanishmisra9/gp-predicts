"""Download practice sessions (laps only) for seasons missing from the FastF1 cache. Needs F1_USERNAME/F1_PASSWORD in env (via scripts_run_ff1.sh)."""
import sys, warnings, fastf1
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache("data/fastf1_cache"); fastf1.set_log_level("ERROR")
for y in [int(a) for a in sys.argv[1:]]:
    sched = fastf1.get_event_schedule(y, include_testing=False)
    for _, ev in sched.iterrows():
        for ses in ("FP1", "FP2", "FP3"):
            try:
                s = ev.get_session(ses); s.load(laps=True, telemetry=False, weather=False, messages=False); print(y, ev.RoundNumber, ses, len(s.laps), flush=True)
            except Exception as e: print("FAIL", y, ev.RoundNumber, ses, type(e).__name__, str(e)[:60], flush=True)
